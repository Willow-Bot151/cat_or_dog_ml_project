import os

def count_files_in_dir(dir_path):
    count = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            count += 1
    return count