import picsellia 
import ignite
import torch
import torchvision 
import os
from datetime import datetime
from typing import List, Tuple
from PIL import Image
from picsellia import Client
from picsellia.sdk.project import Project as PicselliaProject
from picsellia.sdk.experiment import Experiment as PicselliaExperiment
from picsellia.sdk.dataset_version import DatasetVersion as PicselliaDatasetVersion
from picsellia.sdk.deployment import Deployment as DeployedPicselliaModel
import tqdm 
import platform 
import albumentations as A
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2



def get_picsellia_client(organization_name: str = None) -> Client:
    api_token = os.environ.get("PICSELLIA_TOKEN")
    if api_token is None:
        api_token = input("Please enter your TOKEN here :")
    return picsellia.Client(api_token=api_token, organization_name=organization_name)

def checkout_project(client: Client, project_name: str = None) -> PicselliaProject:
    if project_name is None:
        project_name = input("Please enter your project_name here :")
    return client.get_project(project_name)

def get_deployed_model(deployment_name: str = None) -> DeployedPicselliaModel:
    deployment_name = os.environ.get("deployment_name")
    if deployment_name is None:
        deployment_name = input("Please enter your deployment_name here :")
    return get_picsellia_client().get_deployment(deployment_name)


def generate_new_experiment(project: PicselliaProject = None, visualize_valid: bool = False) -> PicselliaExperiment:
    """ 
    This utility function assume that you have at least 3 datasets in your project called
    ('train', 'test', 'valid'),
    This function will: 
       - Create a new experiment 
       - Attach the 3 datasets to this experiment
       - if verbose=True:
            - Will create a fork of the validation set to display the predictions as annotations.
    
    Returns:
        PicselliaExperiment Object
    """
    datasets: List[PicselliaDatasetVersion] = project.list_dataset_versions()
    print(f"{len(datasets)} included in your project:")
    for dataset in datasets:
        print(f"{dataset.name}/{dataset.version}")
        if dataset.version == "train":
            train_dataset = dataset 
        elif dataset.version == "test":
            test_dataset = dataset 
        elif dataset.version == "valid":
            valid_dataset = dataset 
    experiment: PicselliaExperiment = project.create_experiment(
        name=f"v{len(project.list_experiments())} - training"
    )
    experiment.attach_dataset(name="train", dataset_version=train_dataset)
    experiment.attach_dataset(name="test", dataset_version=test_dataset)
    experiment.attach_dataset(name="valid", dataset_version=valid_dataset)

    if visualize_valid: # We will generate a fork of the validation dataset to store the predictions and vizualize them
        valid_dataset_prediction, job = valid_dataset.fork(version=f"valid - prediction - {experiment.name}", assets=valid_dataset.list_assets(), type=valid_dataset.type)
        job.wait_for_done()
        experiment.attach_dataset(name="valid-prediction", dataset_version=valid_dataset_prediction)
    return experiment

def get_train_test_valid_datasets(experiment: PicselliaExperiment) -> Tuple[PicselliaDatasetVersion]:
    train: PicselliaDatasetVersion = experiment.get_dataset('train')
    test : PicselliaDatasetVersion = experiment.get_dataset('test')
    valid: PicselliaDatasetVersion = experiment.get_dataset('valid')
    return (train, test, valid)

def get_transform() -> torchvision.transforms.Compose:
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

def collate_fn(batch):
    return tuple(zip(*batch))
    
class PicselliaCOCODatasetDetection(torch.utils.data.Dataset):
    """ 
    This is a wrapper around Picsellia Dataset version that will download the images and load them into memory.
    """
    def __init__(self, dataset: PicselliaDatasetVersion,
                width: int, 
                height: int, 
                transform: callable = None,
        ) -> None:

        from pycocotools.coco import COCO
        self.dataset = dataset
        self.root = os.path.join(dataset.name, dataset.version)
        self.height = height
        self.width = width 
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.dataset.synchronize(self.root, do_download=True)
        annotations_path = self.dataset.export_annotation_file(
            annotation_file_type=picsellia.types.enums.AnnotationFileType.COCO
        )
        self.coco = COCO(annotations_path)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        if len(target) < 1:
            pass
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB').resize((self.width, self.height))
        if self.transform is not None:
            img = self.transform(img)
        boxes, labels, areas = [], [], []

        num_objs = len(target)

        for i in range(num_objs):
            xmin = target[i]['bbox'][0]
            ymin = target[i]['bbox'][1]
            xmax = xmin + target[i]['bbox'][2]
            ymax = ymin + target[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(target[i]['category_id'])
            areas.append(target[i]['bbox'][2]*target[i]['bbox'][3])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transform is None:
            self.transforms = get_transform()
            img = self.transforms(img)

        return img, my_annotation
    
    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.dataset.name + ' - ' + self.dataset.version  + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list
    if platform.processor() == "arm":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("gpu" if torch.cuda.is_available() else "cpu")


    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(devise) for image in images)
        targets = [{k: v.to(devise) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'outputs/best_model.pth')


def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'outputs/last_model.pth')

def save_loss_plot(OUT_DIR, train_loss, val_loss):
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    figure_2.savefig(f"{OUT_DIR}/valid_loss.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')
    return