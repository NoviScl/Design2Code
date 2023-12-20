import shutil
import os
import json

with open("chenglei_filtered_idx.json") as f:
    filtered_idx = json.load(f)

# Specify the source directory and destination directory
source_dir = '/Users/clsi/Desktop/Pix2Code/testset_cleaned'
dest_dir = '/Users/clsi/Desktop/Pix2Code/testset_final_part1'

# Ensure the destination directory exists, if not, create it
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Copy each file from the source directory to the destination directory
for filename in filtered_idx:
    html = os.path.join(source_dir, str(filename) + '.html')
    png = os.path.join(source_dir, str(filename) + '.png')
    
    if os.path.isfile(html):
        shutil.copy(html, dest_dir)
    if os.path.isfile(png):
        shutil.copy(png, dest_dir)
