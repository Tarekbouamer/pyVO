import sys
import numpy as np
from enum import Enum
import cv2
import os
import glob
import time

from multiprocessing import Process, Queue, Value
from src.utils.sys import Printer


class Dataset(object):
    def __init__(self, path, name, fps=None, associations=None, type=None):
        self.path = path
        self.name = name
        
        self.type = type
        self.fps = fps
        
        if fps is not None:
            self.Ts = 1./fps
        else:
            self.Ts = None
        
        # 
        self.is_ok = True

        self.timestamps = None
        self._timestamp = None       # current timestamp if available [s]
        # next timestamp if available otherwise an estimate [s]
        self._next_timestamp = None

    def __len__(self):
        return len(self.timestamps)
      
    def _load_img(self, frame_id):
        return None

    def getDepth(self, frame_id):
        return None

    def getImageColor(self, frame_id):
        try:
            img = self.getImage(frame_id)
            
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                return img
        except:
            img = None
            Printer.red('Cannot open dataset: ',
                        self.name, ', path: ', self.path)
            return img

    def getTimestamp(self):
        return self._timestamp

    def getNextTimestamp(self):
        return self._next_timestamp
    
    def __getitem__(self, item):

        return None