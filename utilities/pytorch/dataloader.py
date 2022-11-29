import torch 
import torchvision
import picsellia 
import os
from pycocotools.coco import COCO
from PIL import Image
from picsellia.sdk.dataset import DatasetVersion as PicselliaDatasetVersion

def get_transform() -> torchvision.transforms.Compose:
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)
 

class PicselliaCOCODatasetDetection(torch.utils.data.Dataset):
    """ 
    This is a wrapper around Picsellia Dataset version that will download the images and load them into memory.
    It will:
        1) Synchronize your Picsellia dataset locally
        2) Pull the COCO annotation file
        3) Parse the file to implement the get_item() function
    """
    def __init__(self, dataset: PicselliaDatasetVersion,
                width: int, 
                height: int, 
                transform: callable = None,
        ) -> None:


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