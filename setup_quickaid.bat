@echo off
cd /d J:\projects\QuickAid

REM Create the main project structure
mkdir QuickAid
cd QuickAid
mkdir data models results src notebooks

REM Create data folder and subdirectories
mkdir data\images data\labels
mkdir data\images\train data\images\valid data\images\test

REM Add class subfolders inside train, valid, and test
mkdir data\images\train\Bruises data\images\train\Cuts
mkdir data\images\valid\Bruises data\images\valid\Cuts
mkdir data\images\test\Bruises data\images\test\Cuts

REM Add placeholder for YOLOv5 configuration file
echo YOLOv5 configuration file > data\data.yaml

REM Create models folder structure
mkdir models\yolo models\custom
echo Placeholder for YOLOv5 weights > models\yolo\best.pt
echo Placeholder for custom model weights > models\custom\custom_model.pt

REM Create results folder structure
mkdir results\yolo results\custom
mkdir results\yolo\logs results\yolo\outputs
mkdir results\custom\logs results\custom\outputs
echo Placeholder for YOLOv5 metrics > results\yolo\metrics.csv
echo Placeholder for custom model metrics > results\custom\metrics.csv

REM Create source code folder structure
mkdir src\yolo src\custom_model
echo import torch > src\yolo\train_yolo.py
echo import torch > src\yolo\evaluate_yolo.py
echo import torch > src\custom_model\data_loader.py
echo import torch > src\custom_model\simple_model.py
echo import torch > src\custom_model\losses.py
echo import torch > src\custom_model\train_custom.py
echo import torch > src\evaluate.py

REM Create placeholder Jupyter notebook
echo Placeholder for Jupyter Notebook > notebooks\QuickAid.ipynb

REM Create README file
echo # QuickAid Project > README.md

echo Directory structure successfully created!
