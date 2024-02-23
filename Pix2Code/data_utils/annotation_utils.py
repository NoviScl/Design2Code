import os
from collections import defaultdict
import re
import shutil
from PIL import Image, ImageOps
import random
from tqdm import trange

def find_common_png_files(folder_path):
    # Dictionary to hold counts of each file
    file_count = defaultdict(int)
    # Dictionary to hold paths of each file
    file_paths = defaultdict(list)

    # Total number of subfolders
    total_subfolders = 0

    for subdir, dirs, files in os.walk(folder_path):
        if subdir != folder_path:  # Ignore the main folder
            total_subfolders += 1
            for file in files:
                if file.endswith(".png"):
                    file_count[file] += 1
                    file_paths[file].append(os.path.join(subdir, file))

    # Filter out files that are not in all subfolders
    common_files = {file: paths for file, paths in file_paths.items() if file_count[file] == total_subfolders}
    return common_files, file_paths

def rename_and_remove_png_files(folder_path):
    common_files, all_files = find_common_png_files(folder_path)
    print(len(common_files))

    # Extracting numbers from filenames and sorting
    def extract_number(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    sorted_common_files = sorted(common_files.items(), key=lambda x: extract_number(x[0]))

    mapping = {}
    count = 1

    for filename, paths in sorted_common_files:
        for path in paths:
            new_name = f"{count}.png"
            new_path = os.path.join(os.path.dirname(path), new_name)
            os.rename(path, new_path)
        mapping[new_name] = filename
        count += 1

    with open(os.path.join(folder_path, "mapping.txt"), "w") as file:
        for new_path, old_path in mapping.items():
            file.write(f"{new_path}: {old_path}\n")
    
    return count - 1


def remove_files_except(folder_path, count):
    # Iterate over all files in the folder and subfolders
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is not mapping.txt or in the range 1.png to count.png
            if file != 'mapping.txt' and not (file.endswith('.png') and file[:-4].isdigit() and 1 <= int(file[:-4]) <= count):
                full_path = os.path.join(root, file)
                os.remove(full_path)
                print(f"Removed: {full_path}")


def rename_and_move_files(parent_folder):
    # List all subfolders (A, B, C, D, E, F, G) in the parent folder
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

    # Iterate over each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(parent_folder, subfolder)
        
        # List all files in the subfolder
        files = os.listdir(subfolder_path)

        # Iterate over each file
        for file in files:
            # Construct the original file path
            original_file_path = os.path.join(subfolder_path, file)

            # Construct the new file name and path
            new_file_name = f"{subfolder}_{file}"
            new_file_path = os.path.join(parent_folder, new_file_name)

            # Move and rename the file
            shutil.move(original_file_path, new_file_path)

        # Optionally, remove the now-empty subfolder
        os.rmdir(subfolder_path)


def pad_and_resize_image(image, border_size=10):
    # Add a black border around the image
    bordered_img = ImageOps.expand(image, border=border_size, fill='black')

    # Pad the image to make it square
    max_size = max(bordered_img.width, bordered_img.height)
    padded_img = ImageOps.pad(bordered_img, (max_size, max_size), color='white')

    # Resize the image
    resized_img = padded_img.resize((2000, 2000))

    return resized_img

def concatenate_images(folder_path, count):
    # Step 1: Finding all tested approaches
    tested_approaches = set()
    for file in os.listdir(folder_path):
        if file.endswith('.png') and not file.startswith('testset_full') and not file.startswith('gpt4v_direct_prompting'):
            # Split the filename from the extension and then extract the approach name
            file_name = os.path.splitext(file)[0]
            match = re.match(r'(.+)_(\d+)$', file_name)
            if match:
                tested_approach = match.group(1)
                tested_approaches.add(tested_approach)

    # Step 2: Concatenate Images
    for approach in tested_approaches:
        for i in trange(1, count + 1):
            reference_img_path = os.path.join(folder_path, f'testset_full_{i}.png')
            baseline_img_path = os.path.join(folder_path, f'gpt4v_direct_prompting_{i}.png')
            testing_img_path = os.path.join(folder_path, f'{approach}_{i}.png')

            if os.path.exists(reference_img_path) and os.path.exists(baseline_img_path) and os.path.exists(testing_img_path):
                ref_img = pad_and_resize_image(Image.open(reference_img_path))
                baseline_img = pad_and_resize_image(Image.open(baseline_img_path))
                test_img = pad_and_resize_image(Image.open(testing_img_path))

                # Random order
                if random.choice([True, False]):
                    images = [ref_img, baseline_img, test_img]
                    output_filename = f'testset_full_gpt4v_direct_prompting_{approach}_{i}.png'
                else:
                    images = [ref_img, test_img, baseline_img]
                    output_filename = f'testset_full_{approach}_gpt4v_direct_prompting_{i}.png'

                total_width = sum(image.width for image in images)

                # Create a new image with the appropriate size
                new_img = Image.new('RGB', (total_width, 2000))

                x_offset = 0
                for image in images:
                    new_img.paste(image, (x_offset, 0))
                    x_offset += image.width

                # Save the concatenated image
                new_img.save(os.path.join(folder_path, output_filename))


def concatenate_images_with_border(img_paths, output_path, width=400, border_size=2, gap=10, bg_color='white'):
    images = [Image.open(img_path) for img_path in img_paths]

    # Resize images to width while maintaining aspect ratio
    resized_images = [img.resize((width, int(width * img.height / img.width))) for img in images]

    # Add a black border to each image
    bordered_images = [Image.new("RGB", (img.width + 2 * border_size, img.height + 2 * border_size), "black") for img in resized_images]
    for i, img in enumerate(resized_images):
        bordered_images[i].paste(img, (border_size, border_size))

    # Calculate total width for the final image
    total_width = sum(img.width for img in bordered_images) + gap * (len(bordered_images) - 1)
    max_height = max(img.height for img in bordered_images)

    # Create a new image with the correct size
    new_img = Image.new("RGB", (total_width, max_height), bg_color)

    # Paste images into the new image
    x_offset = 0
    for img in bordered_images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width + gap

    # Save the final image
    new_img.save(output_path)


# Replace '/path/to/your/folder' with the path to your main folder
"""
folder_path = '/Users/zhangyanzhe/Downloads/sampled_for_annotation'
count = rename_and_remove_png_files(folder_path)
remove_files_except(folder_path, count)
rename_and_move_files(folder_path)
concatenate_images(folder_path, count)
"""

# Example usage
for i in range(1, 7):
    concatenate_images_with_border([f'/Users/zhangyanzhe/Downloads/example_pix2code/{i}1.png', \
                                    f'/Users/zhangyanzhe/Downloads/example_pix2code/{i}2.png', \
                                    f'/Users/zhangyanzhe/Downloads/example_pix2code/{i}3.png'], \
                                    f'/Users/zhangyanzhe/Downloads/example_pix2code/{i}.png')
