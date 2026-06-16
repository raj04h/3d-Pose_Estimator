import os
import json
import cv2
import torch

from torch.utils.data import Dataset


class satellitePose(Dataset):
    def __init__(self, json_path, img_folder):

        with open(json_path,"r") as f: #reading json file
            self.json_data=json.load(f) 

        self.image_folder=img_folder # store image folder path

    def __len__(self):
        return len(self.json_data)
    

    # Returns one data sample (image + pose label)

    def __getitem__(self, index):
        
        sample=self.json_data[index]  # Access one entry from JSON using index
        filename=sample["filename"] # filename from JSON
 
        # Build full image path
        img_path=os.path.join(self.image_folder, filename)
        image=cv2.imread(img_path)

        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image, (224, 224))

        image=image /255.0 # Normalize the pixel from [0, 2545] to [0, 1]

        # convert image to PyTorch tensor 
        image=torch.tensor(image).permute(2, 0, 1).float()

        # Extract quaternion rotation
        q= torch.tensor(sample["q_vbs2tango"]).float()

        # Extract translation vector
        t=torch.tensor(sample["r_Vo2To_vbs_true"]).float()

        # combine quaternion + translation
        pose=torch.cat((q,t))

        return image, pose