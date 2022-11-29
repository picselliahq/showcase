#%% Relevant imports
import platform
import torch 
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from picsellia_utils import (
    get_picsellia_client,
    checkout_project,
    generate_new_experiment,
    get_train_test_valid_datasets,
    collate_fn,
    save_model,
    PicselliaCOCODatasetDetection,
    Averager,
    SaveBestModel,
    save_loss_plot
)
import time
from picsellia.sdk.experiment import Experiment, ExperimentStatus
from picsellia.types.enums import LogType
from tqdm import tqdm

""" 
This showcase project show you how to perform a Pytorch Object Detection Training. 
Based on Pytorch - Ignite wrapper 
"""
# %% Initializing communication with picsellia and data fetching

# Put your organization name here if you are
# working in an other organization than yours.

# %% Initializing Proper Pytorch Dataloader for training

# Set Batch size



# %% Setting up training

def train(train_data_loader, model, experiment: Experiment):
    print('Training')
    global train_itr
    global train_loss_list
    if platform.processor() == "arm":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device("cpu")
    else:
        device = torch.device("gpu" if torch.cuda.is_available() else "cpu")


     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)

        losses.backward()
        optimizer.step()

        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        experiment.log('Training Loss', data=[float(loss_value)], type=LogType.LINE)
    return train_loss_list

def validate(valid_data_loader, model, experiment: Experiment):
    print('Validating')
    global val_itr
    global val_loss_list
    
    if platform.processor() == "arm":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device("cpu")
    else:
        device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        experiment.log('Validation Loss', data=[float(loss_value)], type=LogType.LINE)
    return val_loss_list

# %%

if __name__ == '__main__':
    client = get_picsellia_client(organization_name=None)
    project = checkout_project(client, project_name="Power Lines Detection")
    experiment = generate_new_experiment(project=project, visualize_valid=False)
    train_ds, test_ds, valid_ds = get_train_test_valid_datasets(experiment=experiment)



    num_epochs = 10
    target_width = 640
    target_height = 640
    train_batch_size = 8

    try:
        experiment_parameters = experiment.get_log("parameters").data
        num_epochs = experiment_parameters.get("num_epochs", num_epochs)
        target_width = experiment_parameters.get("target_width", target_width)
        target_height = experiment_parameters.get("target_height", num_epochs)
        train_batch_size = experiment_parameters.get("batch_size", train_batch_size)
    except Exception as e:
        print("No parameters setup")
    # %% Downloading and generating Dataset for training 

    train_dataset  = PicselliaCOCODatasetDetection(train_ds, target_width, target_height)
    test_dataset   = PicselliaCOCODatasetDetection(test_ds, target_width, target_height)
    valid_dataset  = PicselliaCOCODatasetDetection(valid_ds, target_width, target_height)    

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            collate_fn=collate_fn)
                                            
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            collate_fn=collate_fn)

    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []

    # name to save the trained model with
    MODEL_NAME = experiment.name

    # initialize SaveBestModel class
    save_best_model = SaveBestModel()



    # own DataLoader


    # %% Initializing model for training with pre-trained weights

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(train_ds.list_labels()))
    # %% Finding available device 


    if platform.processor() == "arm":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device("cpu")
    else:
        device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

    model.to(device)
    # %% Initializing the optimizer and criterion

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # start the training epochs
    for epoch in range(num_epochs):
        experiment.update(status=ExperimentStatus.RUNNING)
        print(f"\nEPOCH {epoch+1} of {num_epochs}")

        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model, experiment=experiment)
        val_loss = validate(valid_loader, model, experiment=experiment)
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        # save the best model till now if we have the least loss in the...
        # ... current epoch
        save_best_model(
            val_loss_hist.value, epoch, model, optimizer
        )
        # save the current epoch model
        save_model(epoch, model, optimizer)

        # save loss plot
        save_loss_plot(experiment.name, train_loss, val_loss)
        
        # sleep for 5 seconds after each epoch
        time.sleep(5)
# %%
