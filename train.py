from torchvision.datasets import Cityscapes
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import wandb
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
from pytorch_lightning import seed_everything, LightningModule, Trainer
import multiprocessing
from torchmetrics import JaccardIndex 

def encode_segmap(mask):
    #remove unwanted classes and recitify the labels of wanted classes
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask

def decode_segmap(temp):
    #convert gray scale to color
    temp=temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

class MyClass(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            img_transformed=transform(image)
            seg_transformed=target_transform(target)

            if isinstance(seg_transformed, Image.Image):  # Check if it's still an Image
                seg_transformed = torch.tensor(np.array(seg_transformed))
            seg_transformed = torch.squeeze(seg_transformed)

        return img_transformed, seg_transformed


class OurModel(LightningModule):
    def __init__(self, n_classes, transform):
        super(OurModel, self).__init__()
        # Architecture
        self.layer = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_classes,              # model output channels (number of classes in your dataset)
        )
        
        # Parameters
        self.lr = 1e-3
        self.batch_size = 32
        self.numworker = multiprocessing.cpu_count() // 4
        
        self.criterion = smp.losses.DiceLoss(mode='multiclass')
        self.metrics = JaccardIndex(task='multiclass', num_classes=n_classes)
        
        self.transform = transform
        
        self.train_class = MyClass('./data/', split='train', mode='fine',
                                   target_type='semantic', transforms=transform)
        self.val_class = MyClass('./data/', split='val', mode='fine',
                                 target_type='semantic', transforms=transform)
        
    def forward(self, x):
        return self.layer(x)
    
    def process(self, image, segment):
        out = self(image)
        segment = encode_segmap(segment)
        loss = self.criterion(out, segment.long())
        iou = self.metrics(out, segment)
        return loss, iou
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt
    
    def training_step(self, batch, batch_idx):
        image, segment = batch
        loss, iou = self.process(image, segment)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=False, on_epoch=True, prog_bar=False)

        wandb.log({"train_loss": loss, "train_iou": iou})
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, segment = batch
        loss, iou = self.process(image, segment)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=False)

        wandb.log({"val_loss": loss, "val_iou": iou})
        return loss
    
    def train_dataloader(self):
        return DataLoader(self.train_class, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.numworker, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_class, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.numworker, pin_memory=True)

# Set cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set data
dataset = Cityscapes('./data/', split='train', mode='fine',
                    target_type='semantic')
# 8, 11, 12, 17, 21, 23
ignore_index = 255
class_raw = list(range(34)) + [-1]
'''
valid_classes = [ignore_index,7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
               'train', 'motorcycle', 'bicycle']
colors = [
        [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
'''
target_index = 23
target_class = 'sky'
valid_classes = [ignore_index, target_index]
class_names = ['unlabelled', target_class]
void_classes = [c for c in class_raw if c not in valid_classes]
class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)
colors = [  
    [0, 0, 0],
    [70, 70, 70]
]
label_colours = dict(zip(range(n_classes), colors))

# Set of ImageNet dataset
transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

target_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.RandomHorizontalFlip(),
])

def train():
    # Initialize wandb
    wandb_logger = WandbLogger(
        name='unet-multiclass', 
        project='unet-multiclass'
    )
    dataset = MyClass('./data/', split='train', mode='fine', target_type='semantic', transforms=transform)

    img, seg = dataset[20]

    model = OurModel(n_classes, transform)
    model.to('cuda:0')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath='checkpoints', filename='file',save_last=True)
    # Initialize Trainer
    trainer = Trainer(accelerator="gpu", max_epochs=200, precision=16, callbacks=[checkpoint_callback], logger=wandb_logger)

    # Train the model
    trainer.fit(model)
    torch.save(model.state_dict(), f'./result/{target_class}_model.pth')

if __name__ == "__main__":
    train()
