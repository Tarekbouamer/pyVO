from  src.datasets.generic import  Dataset
import numpy as np
import cv2
import csv

import os


class ArchaeoDataset(Dataset):
    def __init__(self, path, name, fps=10, associations=None, type="kitti"): 
        super().__init__(path, name, fps, associations, type)
        self.timestamps, self.image_names = [], []
        
        # Image sequence
        with open(os.path.join(path, "img_sequence_6.csv")) as file:
            data = csv.reader(file, delimiter=',')
            
            next(data)
            
            for row in data:
                
                self.timestamps.append( row[0]  )
                self.image_names.append(row[1]  )
    
        assert len(self.timestamps) == len(self.image_names)

        self.image_path = os.path.join(path, "images_sequence_6")
        self.max_frame_id = len(self.timestamps)
        self.fps = fps

        print('Processing Archaeo Dataset Sequence of lenght: ', len(self.timestamps))
        
    def _load_img(self, item):
        img_path = None
        img = None

        if item < self.max_frame_id:
            
            try:
                img_path = os.path.join(self.image_path, self.image_names[item])

                img = cv2.imread(img_path)
                self._timestamp = self.timestamps[item]
            
            except:
                print('could not retrieve image: ', item, ' in path ', self.path )
            
            
            if item+1 < self.max_frame_id:   
                self._next_timestamp = self.timestamps[item+1]
            
            else:
                self._next_timestamp = self.timestamps            
                
        return img, self._timestamp, img_path
      
    
    def __getitem__(self, item):
  
        img, time_stamp, path = self._load_img(item)

        height, width, channels = img.shape

        return dict(img=img,
                    time_stamp=time_stamp,
                    idx=item,
                    size=(height, width),
                    path=path
                    )    
      