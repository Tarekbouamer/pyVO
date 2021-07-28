from  src.datasets.generic import  Dataset, GroundTruth
import numpy as np
import cv2
import csv
import sys

import os


class ArchaeoDataset(Dataset):
    def __init__(self, path, name, fps=None, associations=None, type="kitti"): 
        super().__init__(path, name, fps, associations, type)
        self.timestamps, self.image_names = [], []
        

        # Image sequence
        with open(os.path.join(path, "img_sequence_7.csv")) as file:
            data = csv.reader(file, delimiter=',')
            
            next(data)
            
            for row in data:
                
                self.timestamps.append( row[0]  )
                self.image_names.append(row[1]  )
    
        assert len(self.timestamps) == len(self.image_names)

        self.image_path = os.path.join(path, "images_sequence_7")
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
        
        if (item%self.fps) == 0:
            img, time_stamp, path = self._load_img(item)

            height, width, channels = img.shape

            return dict(img=img,
                        time_stamp=time_stamp,
                        idx=item,
                        size=(height, width),
                        path=path
                        )   
        else:
            return None 



class ArchaeoGroundTruth(GroundTruth):
    def __init__(self, path, name, associations=None, type="Archaeo"): 
        super().__init__(path, name, associations, type)
        
        self.x = 0
        self.y = 0
        self.z = 0
        self.scale = 1
        
        self.filename= path   # N.B.: this may depend on how you deployed the groundtruth files 
        
        with open(self.filename) as f:
            self.data = f.readlines()
            self.found = True 
        
        if self.data is None:
            sys.exit('ERROR while reading groundtruth file: please, check how you deployed the files and if the code is consistent with this!') 

    def _getpose(self, frame_id):
        
        if ((frame_id%20)==0):
            line = self._getDataLine(frame_id)
        else:
            line = None
            
        
        if line is not None:
            x = self.scale * float(line[1])
            y = self.scale * float(line[2])
            z = self.scale * float(line[3])     
            scale = 1
            # scale = np.sqrt((x - self.x)*(x - self.x) + 
            #                     (y - self.y)*(y - self.y) + 
            #                     (z - self.z)*(z - self.z))
            
            self.x = x
            self.y = y 
            self.z = z
            self.scale = scale
            
            return x, y, z, scale
        else:
            return self.x, self.y, self.z, self.scale
             
    
    def __getitem__(self, item):

        img, time_stamp, path = self.getpose(item)
        height, width, channels = img.shape

        return dict(img=img,
                    time_stamp=time_stamp,
                    idx=item,
                    size=(height, width),
                    path=path
                    )  