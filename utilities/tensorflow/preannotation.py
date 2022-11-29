import os
from picsellia import Client
from picsellia.sdk.annotation import Annotation
from picsellia.sdk.asset import Asset
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia.sdk.experiment import Experiment
from picsellia.sdk.label import Label
from picsellia.sdk.model_file import ModelFile
from picsellia.sdk.model_version import ModelVersion
from PIL import Image
from picsellia.sdk.project import Project
from tensorflow import saved_model
import numpy as np
from formatter import TensorflowFormatter
from typing import Any, Dict, List
import zipfile


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


#  Initialize Picsellia Client, Project and Experiment
api_token = ""
organization_name = ""

client = Client(api_token=api_token, organization_name=organization_name)


project: Project = client.get_project(PROJECT_NAME)
experiment: Experiment = project.get_experiment(EXPERIMENT_NAME)

trained_weights: ModelFile = experiment.get_artifact("model-latest")

# Download weights from experiment and unzip to have the .pb file
trained_weights.download()
weights_zip_path = trained_weights.filename
with zipfile.ZipFile(weights_zip_path, 'r') as zip_ref:
    zip_ref.extractall("saved_model")
cwd = os.getcwd()
weights_path = os.path.join(cwd,"saved_model")

# Retrieve the test dataset from the exepriment and fork it to create an evaluation version
dataset: DatasetVersion = experiment.get_dataset(TEST_DATASET_NAME)
test_dataset_result_name = f"{dataset.version}-{experiment.name}-results"
assets = dataset.list_assets()
test_result_ds = dataset.fork(
    version=test_dataset_result_name,
    assets=assets,
    type=dataset.type,
)
test_dataset, fork_job = dataset.fork(version=test_dataset_result_name, type=dataset.type)
for label in dataset.list_labels():
    test_dataset.create_label(label.name)
    test_dataset.create_label(label.name+"-predicted")
fork_job.wait_for_done()

# Setup the labels of the new dataset
all_labels = dataset.list_labels()
test_labels = [label for label in all_labels if ("predicted" in label.name) ]

# Retrieve the images 
image_list_path = "images"
dataset.download(image_list_path)
image_list = os.listdir(image_list_path)

model = saved_model.load(weights_path) # Load the model from the weights file

detection_threshold = 0.90 # Set a detection Threshold to avoid the model's noise
for image_path in image_list:
    # Retrieve the asset in the new dataset corresponding to image_path
    origin_asset: Asset = dataset.find_asset(filename=image_path)
    asset: Asset = test_dataset.find_asset(filename=image_path)

    # Load Image and convert to Tensor
    path = os.path.join(image_list_path, image_path)
    image = Image.open(path)
    image_width = image.width
    image_height = image.height
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)

    predictions = model(image)  # Predict
    if len(predictions) > 0:
        #  Format the raw output 
        formatter = TensorflowFormatter(image_width, image_height)
        formated_output = formatter.format_object_detection(predictions)
        scores = formated_output["detection_scores"]
        boxes = formated_output["detection_boxes"]
        classes = formated_output["detection_classes"]

        #  Convert predictions to Picsellia format
        rectangle_list = []
        nb_box_limit = 4
        if len(boxes) < nb_box_limit:
            nb_box_limit = len(boxes)
        for i in range(nb_box_limit):
            if scores[i] >= detection_threshold:
                label: Label = test_labels[int(classes[i])-1]
                box = boxes[i]
                box.append(label)
            rectangle_list.append(tuple(box))
        origin_rectangles = []

        #  Fetch original annotation and shapes to overlay over predictions
        origin_annotation: Annotation = origin_asset.list_annotations()[0]
        annotation: Annotation = asset.create_annotation(duration=0.0)
        for rectangle in origin_annotation.list_rectangles():
            box = [rectangle.x, rectangle.y, rectangle.w, rectangle.h]
            for lb in all_labels:
                if rectangle.label.name == lb.name:
                    matching_label = lb
            box.append(matching_label)
            origin_rectangles.append(tuple(box))
        
        #  Add both the ground truth and predictions to the image in the evaluation dataset
        annotation.create_multiple_rectangles(origin_rectangles)
        annotation.create_multiple_rectangles(rectangle_list)
