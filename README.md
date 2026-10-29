# Faulty Solar Panel Classification (Ensemble CNN)

Classifies solar panel photos into six conditions using an ensemble of three pretrained CNNs with test-time augmentation. Built for a Deep Learning assignment (IA-2).

Classes: Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered.

## Models

- **EfficientNet-B0** with CBAM attention
- **ResNet50** with CBAM attention
- **DenseNet121**

Each model is fine-tuned with Adam and cross-entropy loss with label smoothing. At prediction time the ensemble averages the softmax outputs of all three models on the original image and a flipped copy (TTA).

## Results (validation set, 177 images)

| Model | Accuracy |
|---|---|
| EfficientNet-B0 + CBAM | 90.4% |
| ResNet50 + CBAM | 91.5% |
| DenseNet121 | 89.3% |
| **Ensemble + TTA** | **93.2%** |

## Data and training setup

- `Faulty_solar_panel/`: the training dataset, one folder per class (about 880 images).
- Split 80/20 into train and validation with a fixed seed (100) for reproducibility.
- Training images are augmented with flips, a small rotation and colour jitter, and resized to 224×224 with ImageNet normalisation.
- A `WeightedRandomSampler` corrects for class imbalance; Physical-Damage has the fewest images.
- `Solar_Panel_custom/`: a few extra images per class for checking the model on unseen photos.

## Repository layout

```
cringe.ipynb            Full pipeline: data loading, training, evaluation, ensemble, custom image check
Faulty_solar_panel/     Dataset (ImageFolder layout)
Solar_Panel_custom/     Custom test images
Models/                 Trained weights: efficient_100.pth, resnet100.pth, densenet100.pth
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook cringe.ipynb
```

A CUDA GPU is used when available. To skip training, load the checkpoints from `Models/` and run the ensemble cells.
