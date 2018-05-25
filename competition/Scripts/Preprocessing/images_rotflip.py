# -*- coding: utf-8 -*-
"""
Created on Sat May 19 14:01:34 2018

@author: carlo modica
functions form mouse vs pyth
https://www.blog.pythonlibrary.org/2017/10/05/how-to-rotate-mirror-photos-with-python/
"""
import os
import json
from PIL import Image
import numpy

def rotate(image_path, degrees_to_rotate, saved_location):
    """
    Rotate the given photo the amount of given degreesk, show it and save it
 
    @param image_path: The path to the image to edit
    @param degrees_to_rotate: The number of degrees to rotate the image
    @param saved_location: Path to save the cropped image
    """
   #Image.open(image_path).convert('RGB').save(image_path)
    image_obj = Image.open(image_path)
    rotated_image = image_obj.rotate(degrees_to_rotate)
    rotated_image.save(saved_location)

def flip_image(image_path, saved_location):
    """
    Flip or mirror the image
 
    @param image_path: The path to the image to edit
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(saved_location)

#the new photos id are goint to start from the old id
for file in os.listdir('C:\home\cooluser\competition\Data\Images\Original\Train'):
    try:
        ID=file[:-5]
        
        source=os.path.join('C:\home\cooluser\competition\Data\Images\Original\Train',file)
        
        destination=os.path.join('C:\home\cooluser\competition\Data\Images\Original\Train',ID+'_rot_+15.jpeg')
        rotate(source, 15,destination)
        
        destination=os.path.join('C:\home\cooluser\competition\Data\Images\Original\Train',ID+'_rot_-15.jpeg')
        rotate(source, 345,destination)
        
        destination=os.path.join('C:\home\cooluser\competition\Data\Images\Original\Train',ID+'_flipped.jpeg')
        flip_image(source,destination)
        
    except:
        pass
