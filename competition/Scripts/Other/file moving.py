import json
import urllib3
import shutil 
import numpy
import os
http = urllib3.PoolManager()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

with open('dataset.txt','r') as f:
  furniture = json.load(f)
    
images = furniture['images']
annotations = furniture['annotations']
_id = list()
_label=list()
for i in range(0,len(images)):
   _id.append(images[i]['image_id'])
   _label.append(annotations[i]['label_id'])

_id=numpy.array(_id)
t=1
for file in os.listdir("C:/ass/10000"):
    _id[t]=file[:-5]
    if file.endswith(".jpeg"):
        src_dir = os.path.join("C:/ass/10000/",file)
        dst_dir = os.path.join('C:/ass/LSDA_f/',str(_label[_id[t]]),file)
        shutil.copy(src_dir,dst_dir)
    t=t+1