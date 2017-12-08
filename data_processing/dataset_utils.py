import numpy as np
import metadata_processing as mp
import imageio  
from skimage import transform 
from matplotlib import pyplot as plt
import os
import json

train_image_path = "../data/train/"
test_image_path = "../data/test/"

def load_image(size):
    train_data = []
    test_data = []
    print ("Generating metadata for training.")
    if not os.path.exists('./metadata_train.json'):
        metadata = mp.get_metadata('../data/train/digitStruct.mat')
        with open('metadata_train.json', 'w') as outfile:
            json.dump(metadata, outfile, indent=2)
    else:
        metadata = json.load(open('metadata_train.json'))
        
    meta_train = mp.get_digit_border(metadata)
    print ("Done")
    print ("Load training dataset labels")
    metadata = mp.extend_label(metadata)
    train_labels = np.stack(metadata['label'])
    print ("Done")
    print ()
    
    print ("Generating metadata for testing.")
    if not os.path.exists('./metadata_test.json'):
        metadata = mp.get_metadata('../data/test/digitStruct.mat')
        with open('metadata_test.json', 'w') as outfile:
            json.dump(metadata, outfile, indent=2)
    else:
        metadata = json.load(open('metadata_test.json'))
    meta_test = mp.get_digit_border(metadata)
    print ("Done")
    print ("Load test dataset labels")
    metadata = mp.extend_label(metadata)
    test_labels = np.stack(metadata['label'])
    print ("Done")
    print ()
    print ("loading traning data:")
    for i in range(1, 33403):
        if i % 1000 is 0 or i == 33402:
            print (str(i) + "/" + str(33402))
        try:
            image = imageio.imread(train_image_path + str(i) + ".png")
            chop_image = image[meta_train[i - 1][0]:meta_train[i - 1][1], meta_train[i - 1][2]:meta_train[i - 1][3]]
            '''
            plt.imshow(image, interpolation='nearest')
            plt.show()
            '''
            resized_image = transform.resize(chop_image, size)
        
            train_data.append(resized_image)
        except:
            print (i)
            print (image.shape)
            print (chop_image.shape)
            print (resized_image.shape)
            print ()
            
    print ("loading traning data:")      
    for i in range(1, 13069):
        if i % 1000 is 0 or i == 13068:
            print (str(i) + "/" + str(13068))
        try:
            image = imageio.imread(test_image_path + str(i) + ".png")
            chop_image = image[meta_test[i - 1][0]:meta_test[i - 1][1], meta_test[i - 1][2]:meta_test[i - 1][3]]
            '''
            plt.imshow(image, interpolation='nearest')
            plt.show()
            '''
            resized_image = transform.resize(chop_image, size)
        
            test_data.append(resized_image)
        except:
            print (i)
            print (image.shape)
            print (chop_image.shape)
            print (resized_image.shape)
            print ()
            
            
            
    train_data = np.stack(train_data, axis=0)
    test_data = np.stack(test_data, axis=0)
    return train_data, test_data, train_labels, test_labels

def load_labels():
    if not os.path.exists('./metadata_train.json'):
        metadata = mp.get_metadata('../data/train/digitStruct.mat')
        with open('metadata_train.json', 'w') as outfile:
            json.dump(metadata, outfile, indent=2)
    else:
        metadata = json.load(open('metadata_train.json'))
    
    
    metadata = mp.extend_label(metadata)
    #print (metadata['label'])
    train_labels = np.stack(metadata['label'])    
    
    print ("Generating metadata for testing.")
    if not os.path.exists('./metadata_test.json'):
        metadata = mp.get_metadata('../data/test/digitStruct.mat')
        with open('metadata_test.json', 'w') as outfile:
            json.dump(metadata, outfile, indent=2)
    else:
        metadata = json.load(open('metadata_test.json'))
    metadata = mp.extend_label(metadata)
    test_labels = np.stack(metadata['label'])
    
    return train_labels, test_labels
    
    