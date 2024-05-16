import os
from PIL import Image

data_dir = './data'

def check_truncated_images(directory):
    truncated_images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path[-1] == 'n':
                continue 
            try:
                img = Image.open(file_path)
                img.verify()  # Attempt to open and verify image
            except (IOError, SyntaxError) as e:
                print(f"Error opening image: {file_path} - {e}")
                truncated_images.append(file_path)
    return truncated_images

truncated_images = check_truncated_images(data_dir)

if len(truncated_images) > 0:
    print("\nTruncated images found:")
    for img_path in truncated_images:
        print(img_path)
else:
    print("\nNo truncated images found.")
