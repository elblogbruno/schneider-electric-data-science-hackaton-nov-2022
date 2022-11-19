from dataset import Dataset
import os
from models import ModelManager
from balance import start_balancing
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
        print("[MAIN] Loading dataset...")
        
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
            
    else:
        print("[MAIN] Loading dataset from folder...")
        train_dataset = Dataset(type='csv', file_name='result/train_processed.csv', target_variable_name=target_variable_name)
        img_dataset_train = Dataset(type='img', file_name='train_test_data/train') # Original size
        
    if os.path.exists('result/test_processed.csv') == False or force:
        test_dataset = Dataset(type='csv', file_name='test/test.csv', target_variable_name=target_variable_name)
        
        # categorical_columns = test_dataset.pre_process.get_categorical_columns()
        # test_dataset.pre_process.categorize_data(categorical_columns)      # categorical_columns = ['pollutant']
        
        test_dataset.save_dataset(file_name='result/test_processed.csv')
    else:
        test_dataset = Dataset(type='csv', file_name='result/test_processed.csv', target_variable_name=target_variable_name)

    return train_dataset, test_dataset
    

def save_final_result(test_dataset, model):
    import pandas as pd
    print("[MAIN] Saving final result...")
    pred_y = model.predict(test_dataset.dataset)

    # Empty dataset with two columns (target variable and predicted value)
    final_dataset = pd.DataFrame(columns=['test_index', 'pollutant'])  
    final_dataset['test_index'] = range(len(test_dataset.dataset))
    final_dataset['pollutant'] = pred_y

    final_dataset.to_csv('predictions.csv', index = False, columns=['test_index', 'pollutant']) 
    final_dataset.to_json('predictions.json')



if __name__ == '__main__':
    target_variable_name = 'label'

    train_dataset, test_dataset = load_and_process_data(target_variable_name, force= False)

    print("[MAIN] TRAIN SIZE:" + str(len(train_dataset.dataset)))
    print("[MAIN] TEST_SIZE:" + str(len(test_dataset.dataset)))

    train_dataset.pre_process.show_dataset_correlation_heatmap()

    highest_corrl = train_dataset.pre_process.get_highest_correlation()

    model_manager = ModelManager(train_dataset, test_dataset, target_variable_name)
    model = model_manager.get_best_model()
    save_final_result(test_dataset, model)

    # model_manager.find_best_parameter(model)

    
    # SAVE DATASET
    # file = "models.csv"
    # sm = ManageModels()
    # sm.store_model(model, file)
    
    


    






