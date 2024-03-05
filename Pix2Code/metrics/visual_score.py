import cv2
import numpy as np

# This is a patch for color map, which is not updated for newer version of numpy
def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import random
import os
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from tqdm import tqdm 
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import clip
from copy import deepcopy
from collections import Counter
from Pix2Code.metrics.ocr_free_utils import get_blocks_ocr_free
from Pix2Code.data_utils.dedup_post_gen import check_repetitive_content
from bs4 import BeautifulSoup, NavigableString, Comment
import re
import math
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def calculate_similarity(block1, block2, max_distance=1.42):
    text_similarity = SequenceMatcher(None, block1['text'], block2['text']).ratio()
    return text_similarity


def adjust_cost_for_context(cost_matrix, consecutive_bonus=1.0, window_size=20):
    if window_size <= 0:
        return cost_matrix

    n, m = cost_matrix.shape
    adjusted_cost_matrix = np.copy(cost_matrix)

    for i in range(n):
        for j in range(m):
            bonus = 0
            if adjusted_cost_matrix[i][j] >= -0.5:
                continue
            nearby_matrix = cost_matrix[max(0, i - window_size):min(n, i + window_size + 1), max(0, j - window_size):min(m, j + window_size + 1)]
            flattened_array = nearby_matrix.flatten()
            sorted_array = np.sort(flattened_array)[::-1]
            sorted_array = np.delete(sorted_array, np.where(sorted_array == cost_matrix[i, j])[0][0])
            top_k_elements = sorted_array[- window_size * 2:]
            sum_top_k = np.sum(top_k_elements)
            bonus = consecutive_bonus * sum_top_k
            adjusted_cost_matrix[i][j] += bonus
    return adjusted_cost_matrix

def create_cost_matrix(A, B):
    n = len(A)
    m = len(B)
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = -calculate_similarity(A[i], B[j])
    return cost_matrix


def draw_matched_bboxes(img1, img2, matched_bboxes):
    # Create copies of images to draw on
    img1_drawn = img1.copy()
    img2_drawn = img2.copy()

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    

    # Iterate over matched bounding boxes
    for bbox_pair in matched_bboxes:
        # Random color for each pair
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Ensure that the bounding box coordinates are integers
        bbox1 = [int(bbox_pair[0][0] * w1), int(bbox_pair[0][1] * h1), int(bbox_pair[0][2] * w1), int(bbox_pair[0][3] * h1)]
        bbox2 = [int(bbox_pair[1][0] * w2), int(bbox_pair[1][1] * h2), int(bbox_pair[1][2] * w2), int(bbox_pair[1][3] * h2)]

        # Draw bbox on the first image
        top_left_1 = (bbox1[0], bbox1[1])
        bottom_right_1 = (bbox1[0] + bbox1[2], bbox1[1] + bbox1[3])
        img1_drawn = cv2.rectangle(img1_drawn, top_left_1, bottom_right_1, color, 2)

        # Draw bbox on the second image
        top_left_2 = (bbox2[0], bbox2[1])
        bottom_right_2 = (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3])
        img2_drawn = cv2.rectangle(img2_drawn, top_left_2, bottom_right_2, color, 2)

    return img1_drawn, img2_drawn


def calculate_distance_max_1d(x1, y1, x2, y2):
    distance = max(abs(x2 - x1), abs(y2 - y1))
    return distance


def calculate_ratio(h1, h2):
    return max(h1, h2) / min(h1, h2)


def rgb_to_lab(rgb):
    """
    Convert an RGB color to Lab color space.
    RGB values should be in the range [0, 255].
    """
    # Create an sRGBColor object from RGB values
    rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)
    
    # Convert to Lab color space
    lab_color = convert_color(rgb_color, LabColor)
    
    return lab_color

def color_similarity_ciede2000(rgb1, rgb2):
    """
    Calculate the color similarity between two RGB colors using the CIEDE2000 formula.
    Returns a similarity score between 0 and 1, where 1 means identical.
    """
    # Convert RGB colors to Lab
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    
    # Calculate the Delta E (CIEDE2000)
    delta_e = delta_e_cie2000(lab1, lab2)
    
    # Normalize the Delta E value to get a similarity score
    # Note: The normalization method here is arbitrary and can be adjusted based on your needs.
    # A delta_e of 0 means identical colors. Higher values indicate more difference.
    # For visualization purposes, we consider a delta_e of 100 to be completely different.
    similarity = max(0, 1 - (delta_e / 100))
    
    return similarity


def calculate_current_cost(cost_matrix, row_ind, col_ind):
    return cost_matrix[row_ind, col_ind].sum()


def merge_blocks_wo_check(block1, block2):
    # Concatenate text
    merged_text = block1['text'] + " " + block2['text']

    # Calculate bounding box
    x_min = min(block1['bbox'][0], block2['bbox'][0])
    y_min = min(block1['bbox'][1], block2['bbox'][1])
    x_max = max(block1['bbox'][0] + block1['bbox'][2], block2['bbox'][0] + block2['bbox'][2])
    y_max = max(block1['bbox'][1] + block1['bbox'][3], block2['bbox'][1] + block2['bbox'][3])
    merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    # Average color
    merged_color = tuple(
        (color1 + color2) // 2 for color1, color2 in zip(block1['color'], block2['color'])
    )

    return {'text': merged_text, 'bbox': merged_bbox, 'color': merged_color}


def calculate_current_cost(cost_matrix, row_ind, col_ind):
    return cost_matrix[row_ind, col_ind].tolist()


def find_maximum_matching(A, B, consecutive_bonus, window_size):
    cost_matrix = create_cost_matrix(A, B)
    cost_matrix = adjust_cost_for_context(cost_matrix, consecutive_bonus, window_size)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    current_cost = calculate_current_cost(cost_matrix, row_ind, col_ind)
    return list(zip(row_ind, col_ind)), current_cost, cost_matrix


def remove_indices(lst, indices):
    for index in sorted(indices, reverse=True):
        if index < len(lst):
            lst.pop(index)
    return lst


def merge_blocks_by_list(blocks, merge_list):
    pop_list = []
    while True:
        if len(merge_list) == 0:
            remove_indices(blocks, pop_list)
            return blocks

        i = merge_list[0][0]
        j = merge_list[0][1]
    
        blocks[i] = merge_blocks_wo_check(blocks[i], blocks[j])
        pop_list.append(j)
    
        merge_list.pop(0)
        if len(merge_list) > 0:
            new_merge_list = []
            for k in range(len(merge_list)):
                if merge_list[k][0] != i and merge_list[k][1] != i and merge_list[k][0] != j and merge_list[k][1] != j:
                    new_merge_list.append(merge_list[k])
            merge_list = new_merge_list


def print_matching(matching, blocks1, blocks2, cost_matrix):
    for i, j in matching:
        print(f"{blocks1[i]} matched with {blocks2[j]}, cost {cost_matrix[i][j]}")


def difference_of_means(list1, list2):
    counter1 = Counter(list1)
    counter2 = Counter(list2)

    for element in set(list1) & set(list2):
        common_count = min(counter1[element], counter2[element])
        counter1[element] -= common_count
        counter2[element] -= common_count

    unique_list1 = [item for item in counter1.elements()]
    unique_list2 = [item for item in counter2.elements()]

    mean_list1 = sum(unique_list1) / len(unique_list1) if unique_list1 else 0
    mean_list2 = sum(unique_list2) / len(unique_list2) if unique_list2 else 0

    if mean_list1 - mean_list2 > 0:
        if min(unique_list1) > min(unique_list2):
            return mean_list1 - mean_list2
        else:
            return 0.0
    else:
        return mean_list1 - mean_list2


def find_possible_merge(A, B, consecutive_bonus, window_size, debug=False):
    merge_bonus = 0.0
    merge_windows = 1

    def sortFn(value):
        return value[2]

    while True:
        A_changed = False
        B_changed = False

        matching, current_cost, cost_matrix = find_maximum_matching(A, B, merge_bonus, merge_windows)
        if debug:
            print("Current cost of the solution:", current_cost)
            print_matching(matching, A, B, cost_matrix)
    
        if len(A) >= 2:
            merge_list = []
            for i in range(len(A) - 1):
                new_A = deepcopy(A)
                new_A[i] = merge_blocks_wo_check(new_A[i], new_A[i + 1])
                new_A.pop(i + 1)
    
                updated_matching, updated_cost, cost_matrix = find_maximum_matching(new_A, B, merge_bonus, merge_windows)
                diff = difference_of_means(current_cost, updated_cost)
                if  diff > 0.05:
                    merge_list.append([i, i + 1, diff])
                    if debug:
                        print(new_A[i]['text'], diff)

            merge_list.sort(key=sortFn, reverse=True)
            if len(merge_list) > 0:
                A_changed = True
                A = merge_blocks_by_list(A, merge_list)
                matching, current_cost, cost_matrix = find_maximum_matching(A, B, merge_bonus, merge_windows)
                if debug:
                    print("Cost after optimization A:", current_cost)

        if len(B) >= 2:
            merge_list = []
            for i in range(len(B) - 1):
                new_B = deepcopy(B)
                new_B[i] = merge_blocks_wo_check(new_B[i], new_B[i + 1])
                new_B.pop(i + 1)
    
                updated_matching, updated_cost, cost_matrix = find_maximum_matching(A, new_B, merge_bonus, merge_windows)
                diff = difference_of_means(current_cost, updated_cost)
                if diff > 0.05:
                    merge_list.append([i, i + 1, diff])
                    if debug:
                        print(new_B[i]['text'], diff)

            merge_list.sort(key=sortFn, reverse=True)
            if len(merge_list) > 0:
                B_changed = True
                B = merge_blocks_by_list(B, merge_list)
                matching, current_cost, cost_matrix = find_maximum_matching(A, B, merge_bonus, merge_windows)
                if debug:
                    print("Cost after optimization B:", current_cost)

        if not A_changed and not B_changed:
            break
    matching, _, _ = find_maximum_matching(A, B, consecutive_bonus, window_size)
    return A, B, matching


def merge_blocks_by_bbox(blocks):
    merged_blocks = {}
    
    # Traverse and merge blocks
    for block in blocks:
        bbox = tuple(block['bbox'])  # Convert bbox to tuple for hashability
        if bbox in merged_blocks:
            # Merge with existing block
            existing_block = merged_blocks[bbox]
            existing_block['text'] += ' ' + block['text']
            existing_block['color'] = [(ec + c) / 2 for ec, c in zip(existing_block['color'], block['color'])]
        else:
            # Add new block
            merged_blocks[bbox] = block

    return list(merged_blocks.values())


def mask_bounding_boxes_with_inpainting(image, bounding_boxes):
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Create a black mask
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)

    height, width = image_cv.shape[:2]

    # Draw white rectangles on the mask
    for bbox in bounding_boxes:
        x_ratio, y_ratio, w_ratio, h_ratio = bbox
        x = int(x_ratio * width)
        y = int(y_ratio * height)
        w = int(w_ratio * width)
        h = int(h_ratio * height)
        mask[y:y+h, x:x+w] = 255

    # Use inpainting
    inpainted_image = cv2.inpaint(image_cv, mask, 3, cv2.INPAINT_TELEA)

    # Convert back to PIL format
    inpainted_image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))

    return inpainted_image_pil


def rescale_and_mask(image_path, blocks):
    # Load the image
    with Image.open(image_path) as img:
        # use inpainting instead of simple mask
        img = mask_bounding_boxes_with_inpainting(img, blocks)

        width, height = img.size

        # Determine which side is shorter
        if width < height:
            # Width is shorter, scale height to match the width
            new_size = (width, width)
        else:
            # Height is shorter, scale width to match the height
            new_size = (height, height)

        # Resize the image while maintaining aspect ratio
        img_resized = img.resize(new_size, Image.LANCZOS)

        return img_resized


def calculate_clip_similarity_with_blocks(image_path1, image_path2, blocks1, blocks2):
    # Load and preprocess images
    image1 = preprocess(rescale_and_mask(image_path1, [block['bbox'] for block in blocks1])).unsqueeze(0).to(device)
    image2 = preprocess(rescale_and_mask(image_path2, [block['bbox'] for block in blocks2])).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        image_features1 = model.encode_image(image1)
        image_features2 = model.encode_image(image2)

    # Normalize features
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    similarity = (image_features1 @ image_features2.T).item()

    return similarity


def truncate_repeated_html_elements(soup, max_count=50):
    content_counts = {}

    for element in soup.find_all(True):
        if isinstance(element, (NavigableString, Comment)):
            continue
        
        try:
            element_html = str(element)
        except:
            element.decompose()
            continue
        content_counts[element_html] = content_counts.get(element_html, 0) + 1

        if content_counts[element_html] > max_count:
            element.decompose()

    return str(soup)


def make_html(filename):
    with open(filename, 'r') as file:
        content = file.read()

    if not re.match(r'<html[^>]*>', content, re.IGNORECASE):
        new_content = f'<html><body><p>{content}</p></body></html>'
        with open(filename, 'w') as file:
            file.write(new_content)


def pre_process(html_file):
    check_repetitive_content(html_file)
    make_html(html_file)
    with open(html_file, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')
    soup_str = truncate_repeated_html_elements(soup)
    with open(html_file, 'w') as file:
        file.write(soup_str)


def visual_eval_v3_multi(input_list, debug=False):
    predict_html_list, original_html = input_list[0], input_list[1]
    predict_img_list = [html.replace(".html", ".png") for html in predict_html_list]
    try:
        predict_blocks_list = []
        for predict_html in predict_html_list:
            predict_img = predict_html.replace(".html", ".png")
            # This will help fix some html syntax error
            pre_process(predict_html)
            os.system(f"python3 screenshot_single.py --html {predict_html} --png {predict_img}")
            predict_blocks = get_blocks_ocr_free(predict_img)
            predict_blocks_list.append(predict_blocks)

        original_img = original_html.replace(".html", ".png")
        os.system(f"python3 screenshot_single.py --html {original_html} --png {original_img}")
        original_blocks = get_blocks_ocr_free(original_img)
        original_blocks = merge_blocks_by_bbox(original_blocks)

        # Consider context similarity for block matching
        consecutive_bonus, window_size = 0.1, 1

        return_score_list = []

        for k, predict_blocks in enumerate(predict_blocks_list):
            if len(predict_blocks) == 0:
                print("[Warning] No detected blocks in: ", predict_img_list[k])
                return_score_list.append([0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0)])
                continue
            elif len(original_blocks) == 0:
                print("[Warning] No detected blocks in: ", original_img)
                return_score_list.append([0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0)])
                continue

            if debug:
                print(predict_blocks)
                print(original_blocks)
        
            predict_blocks = merge_blocks_by_bbox(predict_blocks)
            predict_blocks_m, original_blocks_m, matching = find_possible_merge(predict_blocks, deepcopy(original_blocks), consecutive_bonus, window_size, debug=debug)
            
            filtered_matching = []
            for i, j in matching:
                text_similarity = SequenceMatcher(None, predict_blocks_m[i]['text'], original_blocks_m[j]['text']).ratio()
                # Filter out matching with low similarity
                if text_similarity < 0.5:
                    continue
                filtered_matching.append([i, j, text_similarity])
            matching = filtered_matching

            indices1 = [item[0] for item in matching]
            indices2 = [item[1] for item in matching]

            matched_list = []
            sum_areas = []
            matched_areas = []
            matched_text_scores = []
            position_scores = []
            text_color_scores = []
        
            unmatched_area_1 = 0.0
            for i in range(len(predict_blocks_m)):
                if i not in indices1:
                    unmatched_area_1 += predict_blocks_m[i]['bbox'][2] * predict_blocks_m[i]['bbox'][3]
            unmatched_area_2 = 0.0
            for j in range(len(original_blocks_m)):
                if j not in indices2:
                    unmatched_area_2 += original_blocks_m[j]['bbox'][2] * original_blocks_m[j]['bbox'][3]
            sum_areas.append(unmatched_area_1 + unmatched_area_2)
        
            for i, j, text_similarity in matching:
                sum_block_area = predict_blocks_m[i]['bbox'][2] * predict_blocks_m[i]['bbox'][3] + original_blocks_m[j]['bbox'][2] * original_blocks_m[j]['bbox'][3]

                # Consider the max postion shift, either horizontally or vertically
                position_similarity = 1 - calculate_distance_max_1d(predict_blocks_m[i]['bbox'][0] + predict_blocks_m[i]['bbox'][2] / 2, \
                                                        predict_blocks_m[i]['bbox'][1] + predict_blocks_m[i]['bbox'][3] / 2, \
                                                        original_blocks_m[j]['bbox'][0] + original_blocks_m[j]['bbox'][2] / 2, \
                                                        original_blocks_m[j]['bbox'][1] + original_blocks_m[j]['bbox'][3] / 2)
                # Normalized ciede2000 formula
                text_color_similarity = color_similarity_ciede2000(predict_blocks_m[i]['color'], original_blocks_m[j]['color'])
                matched_list.append([predict_blocks_m[i]['bbox'], original_blocks_m[j]['bbox']])
        
                # validation check
                if min(predict_blocks_m[i]['bbox'][2], original_blocks_m[j]['bbox'][2], predict_blocks_m[i]['bbox'][3], original_blocks_m[j]['bbox'][3]) == 0:
                    print(f"{predict_blocks_m[i]} matched with {original_blocks_m[j]}")
                assert calculate_ratio(predict_blocks_m[i]['bbox'][2], original_blocks_m[j]['bbox'][2]) > 0 and calculate_ratio(predict_blocks_m[i]['bbox'][3], original_blocks_m[j]['bbox'][3]) > 0, f"{predict_blocks_m[i]} matched with {original_blocks_m[j]}"
        
                sum_areas.append(sum_block_area)
                matched_areas.append(sum_block_area)
                matched_text_scores.append(text_similarity)
                position_scores.append(position_similarity)
                text_color_scores.append(text_color_similarity)
        
                if debug:
                    print(f"{predict_blocks_m[i]} matched with {original_blocks_m[j]}")
                    print(SequenceMatcher(None, predict_blocks_m[i]['text'], original_blocks_m[j]['text']).ratio())
                    print("text similarity score", text_similarity)
                    print("position score", position_similarity)
                    print("color score", text_color_similarity)
                    print("----------------------------------")
                    pass
            """
            if debug:
                img1 = cv2.imread(predict_img_list[k])
                img2 = cv2.imread(original_img)
                img1_with_boxes, img2_with_boxes = draw_matched_bboxes(img1, img2, matched_list)
            
                plt.figure(figsize=(20, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(img1_with_boxes, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(img2_with_boxes, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
            # """

            if len(matched_areas) > 0:
                sum_sum_areas = np.sum(sum_areas)
        
                final_size_score = np.sum(matched_areas) / np.sum(sum_areas)
                final_matched_text_score = np.mean(matched_text_scores)
                final_position_score = np.mean(position_scores)
                final_text_color_score = np.mean(text_color_scores)
                final_clip_score = calculate_clip_similarity_with_blocks(predict_img_list[k], original_img, predict_blocks, original_blocks)
                final_score = 0.2 * (final_size_score + final_matched_text_score + final_position_score + final_text_color_score + final_clip_score)
                return_score_list.append([sum_sum_areas, final_score, (final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score)])
            else:
                print("[Warning] No matched blocks in: ", predict_img_list[k])
                final_clip_score = calculate_clip_similarity_with_blocks(predict_img_list[k], original_img, predict_blocks, original_blocks)
                return_score_list.append([0.0, 0.2 * final_clip_score, (0.0, 0.0, 0.0, 0.0, final_clip_score)])
        return return_score_list
    except:
        print("[Warning] Error not handled in: ", input_list)
        return [[0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0)] for _ in range(len(predict_html_list))]