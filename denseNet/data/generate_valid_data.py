import pickle
import numpy as np
import PIL 
from PIL import Image

with open("test_batch","rb") as f:
    entry = pickle.load(f,encoding="latin1")
    data = entry["data"]
    data = data.reshape(-1,3,32,32)
    data = data.transpose((0,2,3,1))
    for i in np.random.randint(0,9999,(10,)):
        img = Image.fromarray(data[i])
        img.save(f"tmp_{i}.jpg")

