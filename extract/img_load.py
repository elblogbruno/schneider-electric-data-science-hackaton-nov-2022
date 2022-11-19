import os
import threading
import cv2
import time 
import numpy as np

def get_img_train_dataset(path, img_size=0, black_and_white=False):
    # load images efficiently using threads
    # https://stackoverflow.com/questions/42443936/tensorflow-how-to-load-image-data-efficiently
    
    threads = []

    # get all files in the folder
    files = os.listdir(path)
    files = [os.path.join(path, f) for f in files if os.path.isfile(os.path.join(path, f))]

    # get max number of threads to use cpu efficiently
    cpu_count = os.cpu_count() - 1
    max_threads = min(cpu_count, len(files))

    # split files into chunks. T
    chunk_size = int(len(files) / max_threads) + 1
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

    print("Loading images..." + str(len(files)) + " images found.")
    print("Using " + str(max_threads) + " threads.")
    print("Chunk size: " + str(chunk_size))
    if img_size > 0:
        print("Resizing images to " + str(img_size) + "x" + str(img_size))

    if black_and_white:
        print("Converting images to black and white.")

    start_time = time.time()

    # load images in chunks
    images = []  * max_threads    

    for i in range(max_threads):
        images.append([])
        threads.append(threading.Thread(target=load_image, args=(chunks[i], images[i], img_size, black_and_white)))
        threads[i].start()
    
    # wait for all threads to finish
    for t in threads:
        t.join()

    # merge all images
    images = [item for sublist in images for item in sublist]

    print("Done. " + str(len(images)) + " images loaded in " + str(time.time() - start_time) + " seconds.")

    # make sure images are from same size 
    img_size = images[0].shape
    print("Images are : " + str(img_size))
    for img in images:
        if img.shape != img_size:
            print("Image size mismatch: " + str(img.shape))
            raise Exception('Invalid image size')
    
    # return np.array(images)
    return images
    
def load_image(chunk, images, img_size, black_and_white):
    for f in chunk:
        img = cv2.imread(f)
        if img is not None:
            if img_size > 0:
                img = cv2.resize(img, (img_size, img_size))
            if black_and_white:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # images.append(img.flatten())
            images.append(img)
        else:
            print("Image not found: " + f)