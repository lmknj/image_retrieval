import os
import sys
import time
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import cv2



while True:
    # Read user input
    file_name = input("Enter an image name or type 'quit': ")

    if file_name == 'quit':
        break

    if not os.path.exists(file_name):
        print("Error: '{}' does not exist.".format(file_name))
        continue

# Read an image
    image = cv2.imread(file_name)
    
    cv2.imshow(file_name,image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()






    



    







    