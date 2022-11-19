from dataset import Dataset
import os
from models import ModelManager
from balance import start_balancing
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

        join_data_with_images(img_dataset_train, train_dataset)

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

        join_data_with_images(img_dataset_test, test_dataset)

        print("[MAIN] DATASET JOINED WITH IMAGES")
        test_dataset.save_dataset(file_name='result/test_processed_with_images.csv')

    return train_dataset, test_dataset

def join_data_with_images(img_dataset, dataset):
    # print(img_dataset_train.dataset[0])
    # cv2.imshow('image', img_dataset_train.dataset[0])
    # cv2.waitKey(0)
    labels = dataset.dataset['label'].tolist()
    dataset.dataset = pd.DataFrame(img_dataset.dataset)
    
    # add a new column that each row has the image numpy array
    # img_list = img_dataset.dataset
    dataset.dataset['label'] = labels
    
    # set astype to object so we can save the images as numpy arrays
    # dataset.dataset = dataset.dataset.astype({'images': object})

    # dataset.dataset['images'] = dataset.dataset['images'].astype('object')
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



if __name__ == '__main__':
    target_variable_name = 'label'
    
    train_dataset, test_dataset = load_and_process_data(target_variable_name, force= False)

    print(train_dataset.dataset.head())

    # print("[MAIN] TRAIN SIZE:" + str(len(train_dataset.dataset)))
    # print("[MAIN] TEST_SIZE:" + str(len(test_dataset.dataset)))

    # train_dataset.pre_process.show_dataset_correlation_heatmap()

    # highest_corrl = train_dataset.pre_process.get_highest_correlation()

    model_manager = ModelManager(train_dataset, test_dataset, target_variable_name)
    model = model_manager.get_best_model()
    # save_final_result(test_dataset, model)

    # model_manager.find_best_parameter(model)

    
    # SAVE DATASET
    # file = "models.csv"
    # sm = ManageModels()
    # sm.store_model(model, file)
    
    


    






