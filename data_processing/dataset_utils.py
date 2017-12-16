import numpy as np
import metadata_processing as mp
import tensorflow as tf
import imageio  
from skimage import transform 
from matplotlib import pyplot as plt
from random import randint
import os
import json



class dataset:
    def __init__(self):
        self.train_image_path = "../data/train/"
        self.test_image_path = "../data/test/"
        self.idx = 0
        self.rand_idx_train = np.random.permutation(33402)
        self.rand_idx_test = np.random.permutation(13068)
    def build_batch(self, data, labels, batch_size, is_train, shuffle=False):
        idx = self.idx
        
        rand_idx_train = self.rand_idx_train
        rand_idx_test = self.rand_idx_test
        
        if shuffle is False:
            if is_train is True:
                if idx + batch_size > 33402:
                    batch = np.concatenate((data[idx:33402], data[0:idx + batch_size - 33402]), axis = 0)
                    label = np.concatenate((labels[idx:33402], labels[0:idx + batch_size - 33402]), axis = 0)
                    self.idx = idx + batch_size - 33402
                else:
                    batch = data[idx:idx + batch_size]
                    label = labels[idx:idx + batch_size]
                    self.idx = idx + batch_size
            else:
                if idx + batch_size > 13068:
                    batch = np.concatenate((data[idx:13068], data[0:idx + batch_size - 13068]), axis = 0)
                    label = np.concatenate((labels[idx:13068], labels[0:idx + batch_size - 13068]), axis = 0)
                    self.idx = idx + batch_size - 13068
                else:
                    batch = data[idx:idx + batch_size]
                    label = labels[idx:idx + batch_size]
                    self.idx = idx + batch_size
        else:
            if is_train is True:
                if idx + batch_size > 33402:
                    first_part = rand_idx_train[idx:33402]
                    rand_idx_train = np.random.permutation(33402)
                    second_part = rand_idx_train[0:idx + batch_size - 33402]
                    self.rand_idx_train = rand_idx_train
                    mask = np.concatenate((first_part, second_part), axis = 0)
                    
                    batch = data[mask]
                    label = data[mask]
                    self.idx = idx + batch_size - 33402
                else:
                    batch = data[rand_idx_train[idx:idx + batch_size]]
                    label = labels[rand_idx_train[idx:idx + batch_size]]
                    self.idx = idx + batch_size
            else:
                if idx + batch_size > 13068:
                    first_part = rand_idx_test[idx:13068]
                    rand_idx_test = np.random.permutation(13068)
                    second_part = rand_idx_test[0:idx + batch_size - 13068]
                    self.rand_idx_test = rand_idx_test
                    mask = np.concatenate((first_part, second_part), axis = 0)
                    
                    batch = data[mask]
                    label = data[mask]
                    self.idx = idx + batch_size - 13068
                else:
                    batch = data[rand_idx_test[idx:idx + batch_size]]
                    label = labels[rand_idx_test[idx:idx + batch_size]]
                    self.idx = idx + batch_size
        
        return batch, label
                
                               
        
    def load_image(self, size):
        train_image_path = self.train_image_path
        test_image_path = self.test_image_path
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
                x = randint(1, 10)
                y = randint(1, 10)
                train_data.append(resized_image)
                #train_data.append(resized_image[y:y + 54, x:x + 54])
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
                x = randint(1, 10)
                y = randint(1, 10)

                #train_data.append(resized_image[y:y + 54, x:x + 54])
                train_data.append(resized_image)

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

    def load_labels(self):
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
