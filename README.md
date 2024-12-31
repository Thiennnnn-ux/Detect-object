📋 Overview
This project demonstrates how to train a custom object detection model using YOLOv5. The goal is to detect specific objects in images using a dataset customized for your needs.

🚀 Features
Train YOLOv5 on a custom dataset.
Fine-tune the YOLOv5 model for enhanced accuracy.
Detect and classify objects in images and videos.
🛠 Requirements
Make sure the following are installed:

Python >= 3.7
PyTorch >= 1.7
CUDA (optional but recommended for faster training)
Other Python dependencies listed in requirements.txt
To install the required dependencies:

pip install -r requirements.txt

📂 Dataset Preparation
Prepare your dataset following the YOLOv5 format:

kotlin
Sao chép mã
/dataset
├── images
│   ├── train
│   └── val
├── labels
│   ├── train
│   └── val
images/: Contains training and validation images.
labels/: Contains text files with bounding box annotations in YOLO format.
Update the dataset YAML file (e.g., custom_dataset.yaml):

train: path/to/dataset/images/train
val: path/to/dataset/images/val

nc: <number_of_classes>
names: ['class1', 'class2', ...]
🏋️‍♂️ Training the Model
To start training:

python train.py --img 640 --batch 16 --epochs 50 --data custom_dataset.yaml --weights yolov5s.pt
--img: Image size (default: 640x640).
--batch: Batch size (adjust based on GPU memory).
--epochs: Number of training epochs.
--data: Path to the dataset YAML file.
--weights: Pre-trained weights to start with (e.g., yolov5s.pt).

📊 Results
After training, results such as mAP, precision, and recall will be saved in the runs/train/exp folder. Visualization files and trained weights are also stored here.

🔍 Inference
To test the trained model on new images:

python detect.py --weights runs/train/exp/weights/best.pt --source path/to/images
--weights: Path to the trained weights file.
--source: Path to the input images or videos.

📈 Performance Evaluation
Evaluate the model performance using the validation dataset:

python val.py --weights runs/train/exp/weights/best.pt --data custom_dataset.yaml

📝 Notes
Use a GPU for faster training and inference.
Ensure that the dataset is properly labeled and formatted to avoid errors.
