import numpy as np
import _pickle as pickle
import h5py

metadata = {}


    
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
        