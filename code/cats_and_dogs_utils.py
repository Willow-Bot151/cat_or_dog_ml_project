import os
from PIL import Image

def count_files_in_dir(dir_path):
    count = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            count += 1
    return count

def find_max_image_size(dir_path):
    for _,__,files in os.walk(dir_path):
        files_in_dir = files
    widths = []
    heights = []
    for file in files_in_dir:
        file_path = os.path.join(dir_path,file)
        img_size = Image.open(file_path).size
        widths.append(img_size[0])
        heights.append(img_size[1])
    max_width = max(widths)
    max_height = max(heights)
    return max_width, max_height