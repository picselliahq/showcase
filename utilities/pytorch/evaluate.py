import numpy as np
import cv2
import torch
import glob as glob
import os
import time
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import platform 
import picsellia_utils as utils
import tqdm 
from picsellia.types.enums import AnnotationFileType, ImportAnnotationMode

"""
This code example allows you to perform evaluation and push the results to a different Dataset Version.

The input would be: 
    project_name (str) ->  The name of your project .. :) 
    experiment_name (str) -> The name of your experiment .. :) 
    test_dataset_name (str) -> The name of the Test Dataset you want to evaluate on

"""

PROJECT_NAME=""       # add it 
EXPERIMENT_NAME=""    #
TEST_DATASET_NAME=""  # 



client = utils.get_picsellia_client()
project = utils.checkout_project(client, PROJECT_NAME)
experiment = utils.get_experiment(project, experiment_name=EXPERIMENT_NAME)

test_ds = experiment.get_dataset(TEST_DATASET_NAME)

if os.path.isdir(os.path.join(test_ds.name, test_ds.version)):
    test_ds.synchronize(os.path.join(test_ds.name, test_ds.version), do_download=True)
else:
    test_ds.download(os.path.join(test_ds.name, test_ds.version))

print(f"Cloning Dataset: {test_ds.name}/{test_ds.version}")

# Initializing The model 
assets = test_ds.list_assets()
test_result_ds, fork_job = test_ds.fork(
    version=f"{test_ds.version}-{experiment.name}-results",
    assets=assets,
    type=test_ds.type,
)
fork_job.wait_for_done()

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(test_ds.list_labels()))
COLORS = np.random.uniform(0, 255, size=(len(test_ds.list_labels()), 3))
CLASSES = [e.name for e in test_ds.list_labels()]

# %% Finding available device 


if platform.processor() == "arm": # It's a security measure as Pytorch is not managin well the arm 
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

experiment.get_artifact('best_model.pth').download() 
checkpoint = torch.load('best_model.pth', map_location=device)


model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()
# directory where all the images are present
DIR_TEST = os.path.join(test_ds.name, test_ds.version)
test_images = glob.glob(f"{DIR_TEST}/*.jpg")
print(f"Test instances: {len(test_images)}")
# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.8
# to count the total number of images iterated through
frame_count = 0
# to keep adding the FPS for each image
total_fps = 0 
for i in tqdm.tqdm(range(len(test_images))):
    # get the image file name for saving output later on
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    test_asset = test_ds.find_asset(filename=image_name)
    test_result_asset = test_result_ds.find_asset(filename=image_name)
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image.to(device))
    end_time = time.time()
    # get the current fps
    fps = 1 / (end_time - start_time)
    # add `fps` to `total_fps`
    total_fps += fps
    # increment frame count
    frame_count += 1
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    print(outputs)
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        annotation = test_result_ds.create_annotation(duration=0.0)
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        shapes = []
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            label = test_result_ds.get_or_create_label(class_name)
            x1, y1, x2, y2 = int(box[0]), int(box[1]),int(box[2]), int(box[3])
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            shapes.append((x, y, w, h, label))
        annotation.create_multiple_rectangles(shapes)

    print(f"Image {i+1} done...")
    print('-'*50)
print(f"TEST PREDICTIONS COMPLETE and results pushed to {test_ds.version}-{experiment.name}-results")
print("Now pushing the ground truth for comparaison ...")
gtpath = test_ds.export_annotation_file(AnnotationFileType.COCO)
test_result_ds.import_annotations_coco_file(
   file_path=gtpath, mode=ImportAnnotationMode.REPLACE,
   fail_on_asset_not_found=False, force_create_label=True
)
# calculate and print the average FPS