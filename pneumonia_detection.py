import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from keras.preprocessing.image import ImageDataGenerator

path_in = 'C:/Users/Jussi/Documents/python/chest_xray'
dir_train = '%s/train/' % path_in
dir_test = '%s/test/' % path_in
dir_val = '%s/val/' % path_in

train_norm = '%s/NORMAL/' % dir_train
train_pneu = '%s/PNEUMONIA/' % dir_train

# find all training images (already converted from DICOM to jpeg)
imgs_train_norm = glob('%s/*.jpeg' % train_norm)
imgs_train_pneu = glob('%s/*.jpeg' % train_pneu)

# helper function to plot n images from both NORMAL and PNEUMONIA for visual inspection
def show_imgs(n_imgs):
    for i in range(n_imgs):
        img_norm = np.asarray(plt.imread(imgs_train_norm[i]))
        img_pneu = np.asarray(plt.imread(imgs_train_pneu[i]))

        fig = plt.figure(figsize = (15,10))

        plot_norm = fig.add_subplot(1,2,1)
        plt.imshow(img_norm, cmap='gray')
        plot_norm.set_title('Normal')
        plt.axis('off')

        plot_pneu = fig.add_subplot(1,2,2)
        plt.imshow(img_pneu, cmap='gray')
        plot_pneu.set_title('Pneumonia')
        plt.axis('off')

        plt.show()


def preprocessing(dataset, target_size, **kwargs):
    """
        Preprocess data (i.e., generate batches of tensor image data with real-time data augmentation)
        See https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator for additional parameters (**kwargs)
        
        Parameters
        ----------
        dataset : str
            Dataset to use. Options are 'train' (training data), 'test' (testing data), and 'val' (validation data)
        **kwargs : dict
            Keyword arguments that are forwarded to ImageDataGenerator
        """
    datagen = ImageDataGenerator(**kwargs)
    match dataset:
        case 'train':
            dir = dir_train
        case 'test':
            dir = dir_test
        case 'val':
            dir = dir_val
    
    preprocessed_set = datagen.flow_from_directory(dir, target_size=target_size, color_mode='grayscale', batch_size=32, class_mode='binary')   
    return preprocessed_set
