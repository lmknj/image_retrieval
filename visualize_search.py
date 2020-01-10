

import os
import sys
import time
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import pickle

#from mrcnn2 import utils
#import mrcnn2.model as modellib
from mrcnn2 import visualize

# Import COCO config
#import coco

import IoU_getter
from IoU_getter import Get_IoU
import vector_getter
from vector_getter import Get_vector

import copy

import loadAndComparison as Lac
from loadAndComparison import ImageData
from loadAndComparison import LoadAndComparison

from visualize_IoU import Visualize_IoU
import glob

from search_sort import Search_sort

class Visualize_search:
    def __init__(self, search_image):
        self.search_sort = Search_sort(search_image)
        self.result = self.search_sort.sort()



        return
    
    def visualize_results(self):



        return


def main():
    visualize_search = Visualize_search(search_image = "sample.jpg.pickle")
    visualize_search.visualize_results()
    
    return

main()