import os
from torchvision.datasets import Cityscapes
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from train import n_classes, transform, OurModel, MyClass, encode_segmap, decode_segmap

def test(target_class: str, df: pd.DataFrame):
    model = OurModel(n_classes, transform)
    model.load_state_dict(torch.load(f'./result/{target_class}_model.pth'))

    test_class = MyClass('./data/', split='val', mode='fine',
                         target_type='semantic', transforms=transform)
    test_loader = DataLoader(test_class, batch_size=12, shuffle=False)

    model = model.cuda()
    model.eval()

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            img, seg = batch
            output = model(img.cuda())

            for i in range(img.size(0)):
                invimg = inv_normalize(img[i])
                outputx = output.detach().cpu()[i]
                encoded_mask = encode_segmap(seg[i].clone())  # (256, 512)
                decoded_mask = decode_segmap(encoded_mask.clone())  # (256, 512)
                decoded_output = decode_segmap(torch.argmax(outputx, 0))

                # Get the title (filename) of the input image
                image_title = test_class.images[batch_idx * img.size(0) + i].split('_')[0]
                print(f"image_title: {image_title}")

                x_shape = decoded_mask.shape[0]
                y_shape = decoded_mask.shape[1]

                ratio = len(np.nonzero(decoded_output)[0])/(x_shape * y_shape)
                print(f"{image_title}: {ratio}")

                df.loc[image_title, target_class] = ratio

                fig, ax = plt.subplots(ncols=3, figsize=(16, 5), facecolor='white')
                ax[0].imshow(np.moveaxis(invimg.numpy(), 0, 2))  # (3, 256, 512)
                ax[1].imshow(decoded_mask)  # (256, 512, 3)
                ax[2].imshow(decoded_output)  # (256, 512, 3)
                ax[0].axis('off')
                ax[1].axis('off')
                ax[2].axis('off')
                ax[0].set_title('Input Image')
                ax[1].set_title('Ground mask')
                ax[2].set_title('Predicted mask')

                plt.savefig(f'./output/result_batch{batch_idx}_{target_class}_img{i}.png', bbox_inches='tight')
                plt.close(fig)
                break
            
            break
    
    df.to_csv(f"./output/result/result.csv", na_rep='Unknown')

    


if __name__ == "__main__":
    class_list = ['building', 'pole', 'sidewalk', 'sky', 'vegetation', 'wall']

    image_folder = './data/leftImg8bit/val/'
    image_id = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            # Get the full path of the file
            file_path = file
            # file_path = file.split('_')[0]
            image_id.append(file_path)

    image_id = list(set(image_id))
    print(image_id)

    df = pd.DataFrame(columns=class_list, index=image_id)
    for target_class in class_list:
        test(target_class, df)