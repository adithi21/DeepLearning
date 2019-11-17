from .model import getModel
from .data import Cifar10Data
from pathlib import Path
import PIL
import torch

class Inference:
    def __init__(self, model_to_use = 'densenet'):

        save_model_filename="saved_weights.pt"
        
        self.data = Cifar10Data()
        self.model = getModel(training=False,model_to_use=model_to_use,num_classes=self.data.num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(f"./src/saved_weights/"+save_model_filename,map_location=device))
        return None

    def __call__(self,image):
        if isinstance(image,(Path,str)):
            image = PIL.Image.open(image).convert("RGB")
        elif not isinstance(image,PIL.JpegImagePlugin.JpegImageFile): 
            raise Exception("must be PIL image or path ")
        image_input = self.data.val_transforms(image).unsqueeze(0)
        with torch.no_grad():
            out = self.model(image_input)#.squeeze(0)
            prob = torch.argmax(out).item()
            return self.data.trainset.classes[prob]

