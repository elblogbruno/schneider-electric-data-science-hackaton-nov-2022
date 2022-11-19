import torch
from dataset import Dataset
import os

from models import ModelManager
from balance import start_balancing
from performance import Performance
from resnet import load_and_train_model, test_model
from store_models import StoreModels
import cv2

# TAREAS
# 1. Cargar Datos 
    # 1.1 Cargar Datos de Imagenes des de el csv 

# 2. Procesar Datos    
    # 2.1 Ver si el dataset tiene alguna columna con valores nulos o vacios
    # 2.4 Ver si el dataset esta desbalanceado
        # 2.4.1 Arreglarlo 
    # 2.5 Ver si hay que categorizar los datos 
    # 2.6 Ver si hay que normalizar los datos
    # 2.7 Ver variables inutiles (correlacion entre variables) 
    
    
# 3 Entrenar Modelos
    # KNN 
    # SVM
    # Random Forest 
    # Naive Bayes 
    # Decision Tree
    # Logistic Regression 

# 4 Calcular performance de los modelos
    # 4.1 F1-Score 
    # 4.2 Precision 
    # 4.3 Recall 
    # 4.4 Accuracy 

# 5 Guardar Modelos 
    # 5.1 Guardar Graficas
        # 5.2 ROC Curve 
        # 5.3 Precision-Recall Curve 
        # 5.4 Confusion Matrix
    # 5.2 GUARDAR .CSV amb els resultats de predir per a cadascuna de les files de test_x.csv

# label: In this column you will have the following categories

# 'Plantation':Encoded with number 0, Network of rectangular plantation blocks, connected by a well-defined road grid. In hilly areas the layout of the plantation may follow topographic features. In this group you can find: Oil Palm Plantation, Timber Plantation and Other large-scale plantations.
# 'Grassland/Shrubland': Encoded with number 1, Large homogeneous areas with few or sparse shrubs or trees, and which are generally persistent. Distinguished by the absence of signs of agriculture, such as clearly defined field boundaries.
# 'Smallholder Agriculture': Encoded with number 2, Small scale area, in which you can find deforestation covered by agriculture, mixed plantation or oil palm plantation.



def load_and_process_data(target_variable_name, force = False):
    if os.path.exists('result/train_processed.csv') == False or force:
        print("[MAIN] First time executing...")
        
        train_dataset = Dataset(type='csv', file_name='train/train.csv', target_variable_name=target_variable_name)
        # train_dataset.dataset es un DataFrame de pandas con el csv

        print("[MAIN] DATASET BALANCED")
        print(train_dataset.pre_process.get_dataset_balance(train_dataset.dataset))
            

        # img_dataset_train = Dataset(type='img', file_name='train_test_data/train', img_size=256) # 256x256
        # img_dataset_train = Dataset(type='img', file_name='train_test_data/train', img_size=256, black_and_white=True) # 256x256 black and white
        img_dataset_train = Dataset(type='img', file_name='train_test_data/train') # Original size
        
        # do data augmentation
        train_dataset, img_dataset_train = start_balancing(train_dataset, img_dataset_train, type='oversampling')

        print("[MAIN] Finally is DATASET BALANCED?")
        print(train_dataset.pre_process.get_dataset_balance(train_dataset.dataset)) # should be balanced

        # save the dataset augmented with the images so we don't have to do it again
        train_dataset.save_dataset(file_name='result/train_processed.csv')

        # train_dataset.process_data() # process the data and check if there are any null values

        print(len(train_dataset.dataset))
        print(len(img_dataset_train.dataset))
            
    else:
        print("[MAIN] Loading dataset from folder...")
        train_dataset = Dataset(type='csv', file_name='result/train_processed.csv', target_variable_name=target_variable_name)
        img_dataset_train = Dataset(type='img', file_name='train_test_data/train') # Original size

    print("[MAIN] Loading test dataset...")
    
    print("[MAIN] Loading test dataset from folder...")
    test_dataset = Dataset(type='csv', file_name='result/test_processed.csv', target_variable_name=target_variable_name)
    img_dataset_test = Dataset(type='img', file_name='train_test_data/test') # Original size

    return train_dataset, test_dataset, img_dataset_train, img_dataset_test

def join_data_with_images(img_dataset, dataset):
    # add a new column with the images numpy arrays to the dataset
    # train_dataset.dataset.insert(0, 'images', img_dataset_train.dataset.tolist(), True)
    # img_list = img_dataset.dataset
    # print(len(img_list))

    # dataset.dataset.loc[:, 'images'] = img_list
    # dataset.dataset = dataset.dataset.astype({'images': object})

    # print("[MAIN] DATASET JOINED WITH IMAGES")
    # return img_dataset, dataset

    # calculate pixel descriptors for each image
    for i in range(len(img_dataset.dataset)):
        img = img_dataset.dataset[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256))
        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.Canny(img, 100, 200)
        img = cv2.resize(img, (256, 256))
        cv2.imwrite('result/imgs/' + str(i) + '.jpg', img)
        img_dataset.dataset[i] = img.flatten()

    img_list = img_dataset.dataset
    print(len(img_list))

    dataset.dataset.loc[:, 'images'] = img_list
    dataset.dataset = dataset.dataset.astype({'images': object})

    dataset.remove_headers(['latitude', 'longitude', 'year'])

    print("[MAIN] DATASET JOINED WITH IMAGES")
    return img_dataset, dataset

def save_final_result(pred_y):
    import pandas as pd
    print("[MAIN] Saving final result...")
    dic = { "target" : {} }

    for i in range(len(pred_y)):
        dic['target'][i] = int(pred_y[i])

    import json
    # SAVE to .json file
    with open('predictions.json', 'w') as fp:
        json.dump(dic, fp)

    
"""
DEPRECATED CODE
"""
def get_features(img_dataset_train, img_size=332, name='train', black_and_white = False):
    import numpy as np
    name_file = name+'_features.npy'
    
    if (os.path.exists(name_file) == True):
        print("[MAIN] Loading features from folder...")
        features  = np.load(name_file)
        return features

    from torchvision  import transforms
    # Transform the image, so it becomes readable with the model
    # transform = transforms.Compose([
    #     transforms.ToTensor()                              
    # ])

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.CenterCrop(512),
        # transforms.Resize(img_size),
        transforms.ToTensor()                              
    ])

    # Will contain the feature
    features = []

    from feature_extractor import new_model, device

    # Iterate each image
    for i in img_dataset_train.dataset:
        # Transform the image
        img = transform(i)
        
        # Reshape the image. PyTorch model reads 4-dimensional tensor
        # [batch_size, channels, width, height]
        
        img = img.reshape(1, 3, img_size, img_size)
        img = img.to(device)
        # We only extract features, so we don't need gradient
        with torch.no_grad():
            # Extract the feature from the image
            feature = new_model(img)
            print(feature)
        
        # Convert to NumPy Array, Reshape it, and save it to features variable
        features.append(feature.cpu().detach().numpy().reshape(-1))

    # Convert to NumPy Array
    features = np.array(features)

    # Save the features
    np.save(name_file, features)

    return features

if __name__ == '__main__':
    target_variable_name = 'label'

    train_dataset, test_dataset, img_dataset_train, img_dataset_test = load_and_process_data(target_variable_name, force= False)
    
    # we load the data into pytorch tensors and train the resnet model
    model = load_and_train_model(train_dataset, target_variable_name, epochs = 20)

    labels_pred = test_model(img_dataset_test)

    save_final_result(labels_pred)