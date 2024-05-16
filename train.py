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
# 17
ignore_index = 255
class_raw = list(range(34)) + [-1]
valid_classes = [ignore_index, 17]
class_names = ['unlabelled', 'pole']
void_classes = [c for c in class_raw if c not in valid_classes]
class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)
colors = [  
    [0, 0, 0],
    [111, 74, 0]
]
label_colours = dict(zip(range(n_classes), colors))

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
    torch.save(model.state_dict(), './result/pole_model.pth')

if __name__ == "__main__":
    train()

# # Test
# model.load_state_dict(torch.load('./result/model.pth'))

# test_class = MyClass('./data/', split='val', mode='fine',
#                    target_type='semantic',transforms=transform)  
# test_loader=DataLoader(test_class, batch_size=12, 
#                       shuffle=False)

# model=model.cuda()
# model.eval()
# with torch.no_grad():
#     for batch in test_loader:
#         img,seg=batch
#         output=model(img.cuda())
#         break
# print(img.shape,seg.shape,output.shape)

# inv_normalize = transforms.Normalize(
#     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#     std=[1/0.229, 1/0.224, 1/0.255]
# )
# sample=8
# invimg=inv_normalize(img[sample])
# outputx=output.detach().cpu()[sample]
# encoded_mask=encode_segmap(seg[sample].clone()) #(256, 512)
# decoded_mask=decode_segmap(encoded_mask.clone())  #(256, 512)
# decoded_ouput=decode_segmap(torch.argmax(outputx,0))
# fig,ax=plt.subplots(ncols=3,figsize=(16,50),facecolor='white')  
# ax[0].imshow(np.moveaxis(invimg.numpy(),0,2)) #(3,256, 512)
# #ax[1].imshow(encoded_mask,cmap='gray') #(256, 512)
# ax[1].imshow(decoded_mask) #(256, 512, 3)
# ax[2].imshow(decoded_ouput) #(256, 512, 3)
# ax[0].axis('off')
# ax[1].axis('off')
# ax[2].axis('off')
# ax[0].set_title('Input Image')
# ax[1].set_title('Ground mask')
# ax[2].set_title('Predicted mask')
# plt.savefig('result.png',bbox_inches='tight')
