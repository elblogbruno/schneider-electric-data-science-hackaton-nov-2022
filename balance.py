import imblearn
import pandas as pd
from augmentation import rotate_image
import numpy as np
import cv2
import uuid # for generating unique id

def start_balancing(train_dataset, train_images, type='oversampling', augment_images=False):
    print("Start balancing with " + type)
    if type == 'oversampling':
        return oversampling(train_dataset, train_images, augment_images)
    elif type == 'undersampling':
        return undersampling(train_dataset, train_images)
    elif type == 'smote':
        return smote(train_dataset, train_images)
    else:
        raise Exception('Invalid type')

def img_augmentation(X, y, X_res, y_res, train_images, train_dataset):
    # get new added X and y
    new_X = X_res[X_res.index.isin(X.index) == False]
    new_y = y_res[y_res.index.isin(y.index) == False]

    # get 'example_path' column from X that label is X
    dic = {}
    for i in range(3): # 3 is the number of classes in this dataset
        dic[i] = []
        arr = X[y == i]
        dic[i] = arr['example_path'].values
    
    # we get new added classes and for each type we randomly rotate the image and add it to the dataset
    for index, row in new_X.iterrows():
        print(row, int(index))
        label = new_y[index]
        print(label)
        
        # get image from train_images whose label is label
        random_index = np.random.randint(0, len(dic[label]))
        img_path = dic[label][random_index] # get image path
        img = cv2.imread(img_path)

        # rotate image
        img = rotate_image(img, np.random.randint(0, 360))

        train_images.dataset = np.append(train_images.dataset, [img], axis=0) # add image to dataset
        path_to_img = "train_test_data/train/" + str(index) + "_" + str(uuid.uuid4()) + "_" + str(label) + '_rotated.png' # change path to new image

        # modify the new_X
        X_res.at[index, 'example_path'] = path_to_img
        
        # save image to disk
        cv2.imwrite(path_to_img, img)

    train_dataset.dataset = pd.concat([X_res, y_res], axis=1)

    return train_dataset, train_images
    
def oversampling(train_dataset, train_images, augment_images):
    print("Oversampling...")
    oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='all')

    X = train_dataset.dataset.drop(train_dataset.get_target_variable(), axis=1)
    y = train_dataset.dataset[train_dataset.get_target_variable()]

    X_res, y_res = oversample.fit_resample(X, y)

    # TODO: oversample images
    # do random rotation, flip, etc
    # https://stackoverflow.com/questions/42463172/how-to-apply-a-random-transformation-to-images-in-a-numpy-array

    return img_augmentation(X, y, X_res, y_res, train_images, train_dataset) # augment images

def undersampling(train_dataset, train_images):
    print("Undersampling...")
    undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='all')

    X = train_dataset.dataset.drop(train_dataset.get_target_variable(), axis=1)
    y = train_dataset.dataset[train_dataset.get_target_variable()]

    print(y)

    X_res, y_res = undersample.fit_resample(X, y)

    # get removed X and y
    removed_X = X[X.index.isin(X_res.index) == False]
    removed_y = y[y.index.isin(y_res.index) == False]

    # remove images from train_images
    for index, row in removed_X.iterrows():
        img_path = cv2.imread(row['example_path'])
        train_images.dataset = np.delete(train_images.dataset, np.where(train_images.dataset == img_path), axis=0)
    
    train_dataset.dataset = pd.concat([X_res, y_res], axis=1)

    return train_dataset, train_images

def smote(train_dataset, train_images):
    print("Smote...")
    # smote = imblearn.over_sampling.SMOTE(sampling_strategy='minority')
    # mod_train_dataset, mod_train_images = smote.fit_resample(train_dataset.dataset, train_images.dataset)

    # train_dataset.dataset = mod_train_dataset
    # train_images.dataset = mod_train_images

    # print("Done")
    # return train_dataset, train_images

    