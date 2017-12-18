import numpy as np
import _pickle as pickle
import h5py
def get_metadata(filepath):
    f = h5py.File(filepath)

    metadata= {}
    metadata['height'] = []
    metadata['label'] = []
    metadata['left'] = []
    metadata['top'] = []
    metadata['width'] = []
    
    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(int(obj[0][0]))
        else:
            for k in range(obj.shape[0]):
                vals.append(int(f[obj[k][0]][0][0]))
        metadata[name].append(vals)
    
    for item in f['/digitStruct/bbox']:
        f[item[0]].visititems(print_attrs)
        
    return metadata


def alter(data):
    metadata_dict = {}
    for i in range(0, len(data['label'])):
        metadata_dict[str(i + 1)] = {}
        metadata_dict[str(i + 1)]['label'] = data['label'][i]
        metadata_dict[str(i + 1)]['top'] = data['label'][i]
        metadata_dict[str(i + 1)]['left'] = data['label'][i]
        metadata_dict[str(i + 1)]['width'] = data['label'][i]
        metadata_dict[str(i + 1)]['height'] = data['label'][i]
    
    return metadata_dict

def extend_label(data):
    for i in range(0, len(data['label'])):
        data['label'][i].insert(0, len(data['label'][i]))
        while len(data['label'][i]) <= 6:
            data['label'][i].append(10)
    
    return data

def get_digit_border(metadata):
    ret = []
    for i in range(0, len(metadata['label'])):
        top = max(0, min(metadata['top'][i]))
        left = max(0, min(metadata['left'][i]))
        bot = 0
        right = 0
        for idx in range(len(metadata['top'][i])):
            bot = max(bot, metadata['height'][i][idx] + metadata['top'][i][idx])
            right = max(right, metadata['left'][i][idx] + metadata['width'][i][idx])
        
        ret.append([top, bot, left, right])
    
    return ret
        
            
            
    
