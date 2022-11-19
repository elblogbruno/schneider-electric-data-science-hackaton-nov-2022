from sklearn.neighbors import KNeighborsClassifier
import torch
from dataset import Dataset
import os
from models import ModelManager
from balance import start_balancing
from performance import Performance
from store_models import StoreModels
import cv2
import pandas as pd

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
    if os.path.exists('result/train_processed_with_images.csv') == False or force:
        print("[MAIN] First time executing...")
        
        train_dataset = Dataset(type='csv', file_name='train/train.csv', target_variable_name=target_variable_name)
        # train_dataset.dataset es un DataFrame de pandas con el csv

        # get columns that are strings and categorical
        # categorical_columns = train_dataset.pre_process.get_categorical_columns()
        # train_dataset.pre_process.categorize_data(categorical_columns)     
        # train_dataset.save_dataset(file_name='result/train_processed.csv')
        
        print("[MAIN] DATASET BALANCED")
        print(train_dataset.pre_process.get_dataset_balance(train_dataset.dataset))
            

        # img_dataset_train = Dataset(type='img', file_name='train_test_data/train', img_size=256) # 256x256
        # img_dataset_train = Dataset(type='img', file_name='train_test_data/train', img_size=256, black_and_white=True) # 256x256 black and white
        img_dataset_train = Dataset(type='img', file_name='train_test_data/train') # Original size
        
        # img_dataset_train.dataset seria un array de imagenes 
        # img_dataset_train.dataset[0] seria la primera imagen


        # try to load the images from the csv
        # cv2.imshow('image', img_dataset_train.dataset[0])
        # cv2.waitKey(0)

        # do data augmentation
        train_dataset, img_dataset_train = start_balancing(train_dataset, img_dataset_train, type='oversampling', augment_images=True)

        print("[MAIN] Finally is DATASET BALANCED?")
        print(train_dataset.pre_process.get_dataset_balance(train_dataset.dataset))

        # save the dataset augmented with the images so we don't have to do it again
        train_dataset.save_dataset(file_name='result/train_processed.csv')

        # train_dataset.process_data() # process the data and check if there are any null values

        print(len(train_dataset.dataset))
        print(len(img_dataset_train.dataset))
        # add a new column with the images numpy arrays to the dataset
        # train_dataset.dataset.insert(0, 'images', img_dataset_train.dataset.tolist(), True)
        img_list = img_dataset_train.dataset
        print(len(img_list))

        train_dataset.dataset.loc[:, 'images'] = img_list

        print("[MAIN] DATASET JOINED WITH IMAGES")
        train_dataset.save_dataset(file_name='result/train_processed_with_images.csv')
            
    else:
        print("[MAIN] Loading dataset from folder...")
        train_dataset = Dataset(type='csv', file_name='result/train_processed.csv', target_variable_name=target_variable_name)
        img_dataset_train = Dataset(type='img', file_name='train_test_data/train') # Original size

        train_dataset.save_dataset(file_name='result/train_processed_with_images.csv')
        
        train_dataset.remove_headers(['example_path'])

    print("[MAIN] Loading test dataset...")
    
    if os.path.exists('result/test_processed_with_images.csv') == False or force:
        test_dataset = Dataset(type='csv', file_name='test/test.csv', target_variable_name=target_variable_name)
        
        # categorical_columns = test_dataset.pre_process.get_categorical_columns()
        # test_dataset.pre_process.categorize_data(categorical_columns)      # categorical_columns = ['pollutant']
        img_dataset_test = Dataset(type='img', file_name='train_test_data/test') # Original size
        print(len(test_dataset.dataset))
        print(len(img_dataset_test.dataset))
        # add a new column with the images numpy arrays to the dataset
        # train_dataset.dataset.insert(0, 'images', img_dataset_train.dataset.tolist(), True)
        img_list = img_dataset_test.dataset
        print(len(img_list))

        test_dataset.dataset.loc[:, 'images'] = img_list
        test_dataset.dataset = test_dataset.dataset.astype({'images': object})

        print("[MAIN] DATASET JOINED WITH IMAGES")
        test_dataset.save_dataset(file_name='result/test_processed_with_images.csv')

        # test_dataset.save_dataset(file_name='result/test_processed.csv')
    else:
        print("[MAIN] Loading test dataset from folder...")
        test_dataset = Dataset(type='csv', file_name='result/test_processed.csv', target_variable_name=target_variable_name)
        img_dataset_test = Dataset(type='img', file_name='train_test_data/test') # Original size

    return train_dataset, test_dataset

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



def save_final_result(test_dataset, model):
    import pandas as pd
    print("[MAIN] Saving final result...")
    pred_y = model.predict(test_dataset.dataset)

    # Empty dataset with two columns (target variable and predicted value)
    final_dataset = pd.DataFrame(columns=['test_index', 'label'])  
    final_dataset['test_index'] = range(len(test_dataset.dataset))
    final_dataset['label'] = pred_y

    final_dataset.to_csv('predictions.csv', index = False, columns=['test_index', 'label']) 
    final_dataset.to_json('predictions.json')

def get_features(img_dataset_train, img_size=332, name='train', black_and_white = False):
    name_file = name+'_features.npy'
    
    if (os.path.exists(name_file) == True):
        print("[MAIN] Loading features from folder...")
        features  = np.load(name_file)
        return features

    from torchvision  import transforms
    # Transform the image, so it becomes readable with the model
    transform = transforms.Compose([
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
    # train_dataset, test_dataset = load_and_process_data(target_variable_name, force= False)
    train_dataset = Dataset(type='csv', file_name='result/train_processed.csv', target_variable_name=target_variable_name)
    img_dataset_train = Dataset(type='img', file_name='train_test_data/train') # Original size
    img_dataset_test = Dataset(type='img', file_name='train_test_data/test') # Original size
    
    features = get_features(img_dataset_train, name='train', img_size=332)
    features_test = get_features(img_dataset_test, name='test', img_size=332)
    
    print(features.shape)
    print(features_test.shape)
        
    from sklearn.model_selection import train_test_split

    Train_y = train_dataset.dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(features, Train_y, test_size=0.2, random_state=0)
    
    from sklearn.cluster import KMeans

    # Initialize the model
    model = KNeighborsClassifier()
    # model = KMeans(n_clusters=3, random_state=42)
    

    # Fit the data into the model
    model.fit(X_train, y_train)

    # Extract the labels
    # labels = model.labels_

    # print(labels) # [4 3 3 ... 0 0 0]

    # Predict the labels
    pred_y = model.predict(X_test)

    #print(pred_y) # [4 3 3 ... 0 0 0]

    print(len(pred_y))
    print(len(y_test))

    # accuracy = accuracy_score(test_dataset.dataset['label'], pred_y)
    performance = Performance(y_test, pred_y)
    print(performance.get_performance())




