import os
import numpy as np
from PIL import Image

RECORD_NAME = f"../record/res_record_7.npy" # edit this according to your need.

patches = os.listdir("../patches")
ids = list(set(["_".join(x.split("_")[:2]) for x in patches]))

record = np.load(RECORD_NAME, allow_pickle=True).item()
disc_patches = record["discriminative_patches"]

for id_ in ids:
    ijs = [tuple(name[:-4].split("_")[-2:]) for name in patches \
                  if patches.split("_")[:2] == name]
    is_, js_ = zip(*ijs)
    is_ = list(map(int, is_))
    js_ = list(map(int, js_))
    min_i, max_i = min(is_), max(is_)
    min_j, max_j = min(js_), max(js_)
    useful_matrix_i = np.zeros((max_i-min_i+1, max_j-min_j+1), dtype='uint8')
    for ii, i in enumerate(is_):
        j = js_[ii]
        useful_matrix_i[i-min_i, j-min_j] = 1
    id_disc_patches = [x for x in disc_patches if id_ in x]
    # discriminative
    ijs_disc = [tuple(name[:-4].split("_")[-2:]) for name in disc_patches \
                if patches.split("_")[:2] == name]
    for i, j in ijs_disc:
        useful_matrix_i[i, j] = 2
    
    vis = Image.fromarray(useful_matrix_i)
    vis.putpalette(
        np.array([
            [0, 0, 0], # black (background)
            [255, 255, 255], # white (patch)
            [255, 0, 0] # red (discriminative patch)
        ])
    )
    vis.save(os.path.join("../visualize_disc_patches",id_+".png"))