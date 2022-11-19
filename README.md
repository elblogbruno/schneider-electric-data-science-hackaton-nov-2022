# Schneider-electric-data-science-hackaton Nov 2022

Code for the Nov 2022 Schneider Electric Data Science Hackaton


## Background
Deforestation is the permanent removal of standing forests, which occurs for a variety of reasons and has many devastating consequences. The loss of trees and other vegetation can cause climate change, desertification, soil erosion, fewer crops, flooding, increased greenhouse gases in the atmosphere, and a host of problems for Indigenous people. In the last 13 years, more than 43 million hectares of forest have been devastated in the world, an area the size of California, USA. It is important to stop deforestation, as soon as possible, before the damage is irreversible. There are many ways to fight deforestation. This challenge will consist of using the help of thousands of satellites in space to capture images of the earth's surface in order to detect, as soon as possible, areas in the midst of deforestation and prevent its expansion.

## Participants

-   Bruno Moya - 56
-   Ricard Lopez  - 56
-   Joel Poma - 56

## Results

We have first balanced the dataset by doing oversamplgin among others. This meant, we had to also do data augmentaiton as we were adding new elements to our dataset. For that porpuse we did data augmentation, randomly choosing images from existing classes and randomly rotating them. We did not change the lightning of them as we suposed it would be bad for calculating the features.

This is our result

`{'f1_score': 0.809, 'precision': 0.786, 'recall': 0.833, 'accuracy': 0.714}`

## Code structure

We have all datasets working inside a same Class call dataset. Thus they are all interoperable, and we can add custom functions to make our code cleaner ad our life.

- Augmentation.py  does the data augmentation
- Balance.py does the balancing of the dataset
- Custom Image Dataset is a wrapper for our dtayaset class to work with pytorch efficiently.

## How to execute the demonstration?

Run the main.py example and install required libraries with PIP:

- `pip install -r requirements.txt`
- `python main.py`

The code will automatically efficiently load the train and test images, train the resnet neural network and do some testing.
