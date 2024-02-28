import cv2
import numpy as np
def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import math
import random
import os
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from tqdm import tqdm 
from pathlib import Path
# import pytesseract
from PIL import Image, ImageDraw
import torch
import clip
from copy import deepcopy
from collections import Counter
from copy import deepcopy
from Pix2Code.metrics.ocr_free_utils import get_blocks_ocr_free
from Pix2Code.data_utils.dedup_post_gen import check_repetitive_content
from bs4 import BeautifulSoup, NavigableString, Comment
import re


# pytesseract.pytesseract.tesseract_cmd = '/sailhome/clsi/bin/tesseract'
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def mask_text_cv2(cv2_image):
    # Convert the cv2 image (BGR) to PIL Image (RGB)
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    # Use pytesseract to do OCR on the image
    text_data = pytesseract.image_to_data(pil_image)

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)
    print(text_data.split('\n')[0])

    # Process the OCR data
    for line in text_data.split('\n')[1:]:
        if line.strip() == '':
            continue

        parts = line.split()
        print(parts)
        if len(parts) >= 12:
            x, y, width, height = map(int, parts[6:10])
            # Draw a white rectangle over the detected text
            draw.rectangle([x, y, x + width, y + height], fill="white")

    # Convert PIL Image back to cv2 format (BGR)
    masked_cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return masked_cv2_image

def check_match_images(src_img, web_img, visualize=False):
    # Read the images
    image_b = cv2.imread(web_img)
    image_b = mask_text_cv2(image_b)
    image_a = cv2.imread(src_img)

    # SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    keypoints_a, descriptors_a = sift.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = sift.detectAndCompute(image_b, None)

    # FLANN based matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors_a, descriptors_b, k=2)

    # Keep good matches: Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10: # adjust this threshold

        image_matches = cv2.drawMatches(image_a, keypoints_a, image_b, keypoints_b, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
        src_pts = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Use the homography matrix M to transform the corners of Image A to Image B's plane
        h, w = image_a.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Draw the transformed image on Image B
        image_b_with_a = cv2.polylines(image_b, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # gray = cv2.cvtColor(image_b_with_a, cv2.COLOR_BGR2GRAY)
        if visualize:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.axis('off')
            plt.imshow(image_matches)
            plt.show()

        hb, wb = image_b.shape[:2]
        print(hb, wb)
        print(dst)

        # Extract scale and translation (approximate)
        scale_x = np.linalg.norm(dst[1] - dst[0]) / hb
        scale_y = np.linalg.norm(dst[2] - dst[1]) / wb
        translation = dst[0][0] / np.array([hb, wb])

        print(f"Relative height: {scale_x}, Relative width: {scale_y}")
        print(f"Top-Left Corner Coordinate: {translation}")
        return scale_x, scale_y, translation.tolist()
    else:
        print("Image not found!")
        return None, None, [None, None]

def get_ocr_blocks(image_path):
    # This function will use OCR to extract text blocks and their bounding boxes from an image
    image = cv2.imread(image_path)
    img_h, img_w, _ = image.shape
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    blocks = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0:  # Consider blocks with confidence > 60%
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            text = data['text'][i].strip()
            if len(text) == 0:
                continue
            if w <= 0 or h <= 0:
                continue
            blocks.append({'text': text, 'bbox': (x / img_w, y / img_h, w / img_w, h / img_h)})
    return blocks


def match_blocks(blocks1, blocks2, v_scale=0.1):
    # This function will match blocks between two sets based on text similarity, spatial location, and size similarity
    matched_blocks = []
    max_distance = (1 + v_scale**2)**0.5
    
    for block1 in blocks1:
        best_match = None
        highest_score = 0
        
        for block2 in blocks2:
            # Text similarity
            text_similarity = SequenceMatcher(None, block1['text'], block2['text']).ratio()
            
            if text_similarity > 0.8:  # Text must be similar above a threshold
                
                # Spatial proximity (normalized by image dimensions for example)
                spatial_proximity = 1 - ((block1['bbox'][0] - block2['bbox'][0])**2 + (block1['bbox'][1] * v_scale - block2['bbox'][1] * v_scale)**2)**0.5 / max_distance
                
                # Size similarity
                # size_similarity = 1 - abs(block1['bbox'][2]*block1['bbox'][3] - block2['bbox'][2]*block2['bbox'][3]) / max(block1['bbox'][2]*block1['bbox'][3], block2['bbox'][2]*block2['bbox'][3])

                # Combine the scores with weights as needed
                # combined_score = (text_similarity * 0.6) + (spatial_proximity * 0.2) + (size_similarity * 0.2)
                combined_score = (text_similarity * 0.6) + (spatial_proximity * 0.4)

                print(block2)
                print(combined_score)

                if combined_score > highest_score:
                    highest_score = combined_score
                    best_match = block2

        if best_match:
            matched_blocks.append((block1, best_match))
        
        break
    
    return matched_blocks


def calculate_positional_score(bbox1, bbox2, v_scale=0.1):
    max_distance = (1 + v_scale**2)**0.5

    # Calculate the Euclidean distance between the center points of two bounding boxes
    center1 = (bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2)
    center2 = (bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2)
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] * v_scale - center2[1] * v_scale) ** 2) ** 0.5
    
    # Normalize distance based on a predefined max distance, this value could be tuned
    normalized_distance = min(distance / max_distance, 1)
    
    # Calculate score using exponential decay
    score = 1 - normalized_distance
    
    return score

def group_blocks_by_row(blocks, line_overlap_threshold=0.5):
    """
    Group blocks into rows based on their bounding box y-coordinates.
    Blocks that have y-overlapping bounding boxes within a threshold are considered to be on the same row.

    :param blocks: List of block dictionaries with 'bbox' as one of the keys.
    :param line_overlap_threshold: Threshold for considering blocks to be on the same line (relative to image height).
    :return: A list of lists of blocks, with each inner list representing a row.
    """
    # Sort blocks by the top y-coordinate
    sorted_blocks = sorted(blocks, key=lambda b: b['bbox'][1])
    
    rows = []
    current_row = []
    
    for block in sorted_blocks:
        # If current_row is empty, start a new row with the current block
        if not current_row:
            current_row.append(block)
        else:
            # Compare the current block with the last block in the current row
            last_block_in_row = current_row[-1]
            # Calculate the vertical overlap between the two blocks
            top_y_current = block['bbox'][1]
            bottom_y_last = last_block_in_row['bbox'][1] + last_block_in_row['bbox'][3]
            vertical_overlap = max(0, bottom_y_last - top_y_current)
            
            # If there is enough overlap, add the block to the current row
            if vertical_overlap > line_overlap_threshold * min(last_block_in_row['bbox'][3], block['bbox'][3]):
                current_row.append(block)
            else:
                # Otherwise, the current block starts a new row
                current_row.sort(key=lambda b: (b['bbox'][0]))
                rows.extend(current_row)
                current_row = [block]
    
    # Add the last row if it's not empty
    if current_row:
        current_row.sort(key=lambda b: (b['bbox'][0]))
        rows.extend(current_row)
    
    return rows


def merge_blocks(blocks, line_overlap_threshold=1.5, avg_char_space_ratio=2):
    # Sort blocks by their y-coordinate and then by their x-coordinate
    blocks = group_blocks_by_row(blocks)
    
    merged_blocks = []
    last_block = None

    for block in blocks:
        if last_block is not None:
            # Check vertical overlap; if the y difference is smaller than the height of the block, there is an overlap
            y_diff = abs(block['bbox'][1] + block['bbox'][3] - last_block['bbox'][1])
            height = max(last_block['bbox'][3], block['bbox'][3])

            # Estimate average character width for the last block
            last_block_char_count = max(len(last_block['text'].strip()), 1)  # Avoid division by zero
            avg_char_width_last_block = last_block['bbox'][2] / last_block_char_count

            # Estimate the expected space if there was a single space between the two blocks
            expected_space_width = avg_char_width_last_block * avg_char_space_ratio

            # Calculate actual horizontal gap
            right_edge_last_block = last_block['bbox'][0] + last_block['bbox'][2]
            actual_gap = block['bbox'][0] - right_edge_last_block
            
            # Check if blocks are on the same line and the gap is about the width of a space or less
            if y_diff < height * line_overlap_threshold and actual_gap <= expected_space_width and actual_gap >= 0:
                # Merge the text and extend the bbox
                merged_text = f"{last_block['text']} {block['text']}"
                merged_bbox = (
                    last_block['bbox'][0],  # x-coordinate remains the same
                    min(last_block['bbox'][1], block['bbox'][1]),  # y-coordinate is the upper one
                    right_edge_last_block - last_block['bbox'][0] + actual_gap + block['bbox'][2],  # width
                    max(last_block['bbox'][3], block['bbox'][3])  # height is the taller one
                )

                if merged_bbox[2] < 0:
                    print("ALERT!!!")
                    print(f"{last_block} {block}")
                    print(y_diff, height * line_overlap_threshold, actual_gap, expected_space_width)

                last_block = {'text': merged_text, 'bbox': merged_bbox}
            else:
                # No merge, add the last block to the list
                merged_blocks.append(last_block)
                last_block = block
        else:
            last_block = block

    # Don't forget to add the last block
    if last_block:
        merged_blocks.append(last_block)

    return merged_blocks

def calculate_similarity(block1, block2, max_distance=1.42):
    text_similarity = SequenceMatcher(None, block1['text'], block2['text']).ratio()
    spatial_proximity = 1 - ((block1['bbox'][0] - block2['bbox'][0])**2 + (block1['bbox'][1] - block2['bbox'][1])**2)**0.5 / max_distance
    combined_score = (text_similarity * 1.0) + (spatial_proximity * 0.0)
    return combined_score

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
            """
            # Check left context
            for k in range(1, window_size + 1):
                if i >= k and j >= k:
                    # bonus += consecutive_bonus * (cost_matrix[i-k, j-k] < 0)
                    bonus += consecutive_bonus * cost_matrix[i-k, j-k]
            # Check right context
            for k in range(1, window_size + 1):
                if i + k < n and j + k < m:
                    # bonus += consecutive_bonus * (cost_matrix[i+k, j+k] < 0
                    bonus += consecutive_bonus * cost_matrix[i+k, j+k]
            """
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

def find_maximum_matching(A, B, consecutive_bonus, window_size):
    cost_matrix = create_cost_matrix(A, B)
    cost_matrix = adjust_cost_for_context(cost_matrix, consecutive_bonus, window_size)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))

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

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_distance_max_1d(x1, y1, x2, y2):
    distance = max(abs(x2 - x1), abs(y2 - y1))
    return distance

def calculate_ratio(h1, h2):
    return max(h1, h2) / min(h1, h2)

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def print_stat(alist):
    if len(alist) == 0:
        print("Empty list!")
        return
    print("Mean:", np.mean(alist),"Median:", np.median(alist),"Min:", min(alist),"Max:", max(alist))
    return np.mean(alist), np.median(alist), min(alist), max(alist)

def print_stat_geo(alist):
    if len(alist) == 0:
        print("Empty list!")
        return
    print("Geo Mean:", geo_mean(alist),"Median:", np.median(alist),"Min:", min(alist),"Max:", max(alist))
    return geo_mean(alist), np.median(alist), min(alist), max(alist)


def rescale_short_edge_to_long_edge(image_path):
    # Load the image
    with Image.open(image_path) as img:
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

def calculate_clip_similarity(image_path1, image_path2):
    # Load and preprocess images
    image1 = preprocess(rescale_short_edge_to_long_edge(image_path1)).unsqueeze(0).to(device)
    image2 = preprocess(rescale_short_edge_to_long_edge(image_path2)).unsqueeze(0).to(device)

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


def visual_eval(gpt_img, original_img, print_all=False, ocr_free=True):
    """
    gpt_img: file to image rendered by gpt gen code. Please place the html file with the same name in the same folder.
    original_img: file to image rendered by the original code. Please place the html file with the same name in the same folder.
    print_all: print matched information or not. Default to False.
    ocr_free: using ocr free approach or not. Default to True.
    """

    if ocr_free:
        blocks1 = get_blocks_ocr_free(gpt_img)
        blocks2 = get_blocks_ocr_free(original_img)
        consecutive_bonus, window_size = 0.1, 1
    else:
        blocks1 = get_ocr_blocks(gpt_img)
        blocks2 = get_ocr_blocks(original_img)
        consecutive_bonus, window_size = 0.25, 2

        blocks1 = merge_blocks(blocks1)
        blocks2 = merge_blocks(blocks2)

    matching = find_maximum_matching(blocks1, blocks2, consecutive_bonus, window_size)
    matched_list = []
    # print("Matching pairs:")
    location_score = []
    size_score = []
    for i, j in matching:
        # print(f"{blocks1[i]} matched with {blocks2[j]}")
        matched_list.append([blocks1[i]['bbox'], blocks2[j]['bbox']])
        location_score.append(calculate_distance(blocks1[i]['bbox'][0] + blocks1[i]['bbox'][2], \
                                                blocks1[i]['bbox'][1] + blocks1[i]['bbox'][3], \
                                                blocks2[j]['bbox'][0] + blocks2[j]['bbox'][2], \
                                                blocks2[j]['bbox'][1] + blocks2[j]['bbox'][3]))
        if min(blocks1[i]['bbox'][2], blocks2[j]['bbox'][2], blocks1[i]['bbox'][3], blocks2[j]['bbox'][3]) == 0:
            print(f"{blocks1[i]} matched with {blocks2[j]}")
        assert calculate_ratio(blocks1[i]['bbox'][2], blocks2[j]['bbox'][2]) > 0 and calculate_ratio(blocks1[i]['bbox'][3], blocks2[j]['bbox'][3]) > 0, f"{blocks1[i]} matched with {blocks2[j]}"
        size_score.append(calculate_ratio(blocks1[i]['bbox'][2], blocks2[j]['bbox'][2]) * calculate_ratio(blocks1[i]['bbox'][3], blocks2[j]['bbox'][3]))
    if print_all:
        print(f"Matched: {len(location_score)}")
        print("Location Score:")
        print_stat(location_score)
        print("Size Score:")
        print_stat_geo(size_score)
    

    """
    # For Debugging: show matched blocks
    img1 = cv2.imread(gpt_img)
    img2 = cv2.imread(original_img)
    img1_with_boxes, img2_with_boxes = draw_matched_bboxes(img1, img2, matched_list)
    cv2.imwrite(gpt_img.replace(".png", "_demo.png"), img1_with_boxes)
    cv2.imwrite(original_img.replace(".png", "_demo.png"), img2_with_boxes)
    """

    if len(location_score) > 0:
        matched = len(location_score)
        loc_score = np.mean(location_score)
        s_score = geo_mean(size_score)
        final_score = loc_score / np.sqrt(2) + (1 - 1 / s_score)
        return matched, loc_score, s_score, final_score
    else:
        return 0.0, None, None, None


def sum_of_area(blocks):
    sum = 0
    for block in blocks:
        sum += block['bbox'][2] * block['bbox'][3]
    return sum
    
def visual_eval_v1(gpt_img, original_img, print_all=False, ocr_free=True, debug=False):
    """
    gpt_img: file to image rendered by gpt gen code. Please place the html file with the same name in the same folder.
    original_img: file to image rendered by the original code. Please place the html file with the same name in the same folder.
    print_all: print matched information or not. Default to False.
    ocr_free: using ocr free approach or not. Default to True.
    """

    if ocr_free:
        gpt_html = gpt_img.replace(".png", ".html")
        original_html = original_img.replace(".png", ".html")
        os.system(f"python3 {Path(__file__).parent}/screenshot_single.py --html {gpt_html} --png {gpt_img}")
        os.system(f"python3 {Path(__file__).parent}/screenshot_single.py --html {original_html} --png {original_img}")

        blocks1 = get_blocks_ocr_free(gpt_img)
        blocks2 = get_blocks_ocr_free(original_img)
        consecutive_bonus, window_size = 0.1, 1
    else:
        blocks1 = get_ocr_blocks(gpt_img)
        blocks2 = get_ocr_blocks(original_img)
        consecutive_bonus, window_size = 0.25, 2

        blocks1 = merge_blocks(blocks1)
        blocks2 = merge_blocks(blocks2)

    blocks1_area = sum_of_area(blocks1)
    blocks2_area = sum_of_area(blocks2)
    max_blocks_area = max(blocks1_area, blocks2_area)

    matching = find_maximum_matching(blocks1, blocks2, consecutive_bonus, window_size)
    matched_list = []
    scores = []

    for i, j in matching:
        if debug:
            # print(f"{blocks1[i]} matched with {blocks2[j]}")
            # print(SequenceMatcher(None, blocks1[i]['text'], blocks2[j]['text']).ratio())
            pass

        min_block_area = min(blocks1[i]['bbox'][2] * blocks1[i]['bbox'][3], blocks2[j]['bbox'][2] * blocks2[j]['bbox'][3])
        text_similarity = SequenceMatcher(None, blocks1[i]['text'], blocks2[j]['text']).ratio()
        position_similarity = 1 - calculate_distance(blocks1[i]['bbox'][0] + blocks1[i]['bbox'][2] / 2, \
                                                blocks1[i]['bbox'][1] + blocks1[i]['bbox'][3] / 2, \
                                                blocks2[j]['bbox'][0] + blocks2[j]['bbox'][2] / 2, \
                                                blocks2[j]['bbox'][1] + blocks2[j]['bbox'][3] / 2) / np.sqrt(2)
        matched_list.append([blocks1[i]['bbox'], blocks2[j]['bbox']])

        # validation check
        if min(blocks1[i]['bbox'][2], blocks2[j]['bbox'][2], blocks1[i]['bbox'][3], blocks2[j]['bbox'][3]) == 0:
            print(f"{blocks1[i]} matched with {blocks2[j]}")
        assert calculate_ratio(blocks1[i]['bbox'][2], blocks2[j]['bbox'][2]) > 0 and calculate_ratio(blocks1[i]['bbox'][3], blocks2[j]['bbox'][3]) > 0, f"{blocks1[i]} matched with {blocks2[j]}"

        scores.append(min_block_area * text_similarity * position_similarity / max_blocks_area)
    
    if print_all:
        print(f"Matched: {len(location_score)}")
        print("Score:")
        print_stat(scores)

    if debug:
        img1 = cv2.imread(gpt_img)
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

    if len(scores) > 0:
        matched = len(scores)
        final_score = np.sum(scores)
        return matched, final_score
    else:
        return 0.0, 0.0


def color_similarity(color1, color2):
    # Calculate the Euclidean distance between two RGB colors
    distance = math.sqrt(sum((c2 - c1) ** 2 for c1, c2 in zip(color1, color2)))
    
    # Maximum possible Euclidean distance in RGB space
    max_distance = math.sqrt(3 * (255 ** 2))
    
    # Normalize the distance to a value between 0 and 1
    normalized_distance = distance / max_distance
    
    # Return the similarity as 1 - normalized distance
    return 1 - normalized_distance


import math
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


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


def visual_eval_v2(gpt_img, original_img, print_all=False, ocr_free=True, debug=False):
    """
    gpt_img: file to image rendered by gpt gen code. Please place the html file with the same name in the same folder.
    original_img: file to image rendered by the original code. Please place the html file with the same name in the same folder.
    print_all: print matched information or not. Default to False.
    ocr_free: using ocr free approach or not. Default to True.
    """

    if ocr_free:
        gpt_html = gpt_img.replace(".png", ".html")
        original_html = original_img.replace(".png", ".html")
        os.system(f"python3 {Path(__file__).parent}/screenshot_single.py --html {gpt_html} --png {gpt_img}")
        os.system(f"python3 {Path(__file__).parent}/screenshot_single.py --html {original_html} --png {original_img}")

        blocks1 = get_blocks_ocr_free(gpt_img)
        blocks2 = get_blocks_ocr_free(original_img)
        consecutive_bonus, window_size = 0.1, 1
    else:
        blocks1 = get_ocr_blocks(gpt_img)
        blocks2 = get_ocr_blocks(original_img)
        consecutive_bonus, window_size = 0.25, 2

        blocks1 = merge_blocks(blocks1)
        blocks2 = merge_blocks(blocks2)

    blocks1_area = sum_of_area(blocks1)
    blocks2_area = sum_of_area(blocks2)
    max_blocks_area = max(blocks1_area, blocks2_area)

    matching = find_maximum_matching(blocks1, blocks2, consecutive_bonus, window_size)
    matched_list = []
    scores = []
    size_scores = []
    matched_text_scores = []
    position_scores = []
    text_color_scores = []

    for i, j in matching:
        min_block_area = min(blocks1[i]['bbox'][2] * blocks1[i]['bbox'][3], blocks2[j]['bbox'][2] * blocks2[j]['bbox'][3])
        max_block_area = max(blocks1[i]['bbox'][2] * blocks1[i]['bbox'][3], blocks2[j]['bbox'][2] * blocks2[j]['bbox'][3])
        text_similarity = SequenceMatcher(None, blocks1[i]['text'], blocks2[j]['text']).ratio()
        position_similarity = 1 - calculate_distance(blocks1[i]['bbox'][0] + blocks1[i]['bbox'][2] / 2, \
                                                blocks1[i]['bbox'][1] + blocks1[i]['bbox'][3] / 2, \
                                                blocks2[j]['bbox'][0] + blocks2[j]['bbox'][2] / 2, \
                                                blocks2[j]['bbox'][1] + blocks2[j]['bbox'][3] / 2) / np.sqrt(2)
        # scale to 0.5 ~ 1.0
        text_color_similarity = color_similarity(blocks1[i]['color'], blocks2[j]['color']) * 0.5 + 0.5
        matched_list.append([blocks1[i]['bbox'], blocks2[j]['bbox']])

        # validation check
        if min(blocks1[i]['bbox'][2], blocks2[j]['bbox'][2], blocks1[i]['bbox'][3], blocks2[j]['bbox'][3]) == 0:
            print(f"{blocks1[i]} matched with {blocks2[j]}")
        assert calculate_ratio(blocks1[i]['bbox'][2], blocks2[j]['bbox'][2]) > 0 and calculate_ratio(blocks1[i]['bbox'][3], blocks2[j]['bbox'][3]) > 0, f"{blocks1[i]} matched with {blocks2[j]}"

        scores.append(min_block_area * text_similarity * position_similarity * text_color_similarity / max_blocks_area)
        size_scores.append(min_block_area / max_blocks_area)
        matched_text_scores.append(text_similarity)
        position_scores.append(position_similarity)
        text_color_scores.append(text_color_similarity)

        if debug:
            print(f"{blocks1[i]} matched with {blocks2[j]}")
            print(SequenceMatcher(None, blocks1[i]['text'], blocks2[j]['text']).ratio())
            print("size score", min_block_area / max_blocks_area)
            print("text similarity score", text_similarity)
            print("position score", position_similarity)
            print("color score", text_color_similarity)
            print("----------------------------------")
            pass
    
    if print_all:
        print(f"Matched: {len(location_score)}")
        print("Score:")
        print_stat(scores)

    if debug:
        img1 = cv2.imread(gpt_img)
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

    if len(scores) > 0:
        matched = len(scores)
        final_score = np.sum(scores)
        final_size_score = np.sum(size_scores)
        final_matched_text_score = np.mean(matched_text_scores)
        final_position_score = np.mean(position_scores)
        final_text_color_score = np.mean(text_color_scores)
        final_clip_score = calculate_clip_similarity(gpt_img, original_img)
        return matched, final_score, (final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score)
    else:
        return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0)


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

def find_maximum_matching_v3(A, B, consecutive_bonus, window_size):
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

def sortFn(value):
    return value[2]


def print_matching(matching, blocks1, blocks2, cost_matrix):
    for i, j in matching:
        print(f"{blocks1[i]} matched with {blocks2[j]}, cost {cost_matrix[i][j]}")


def difference_of_means(list1, list2):
    # Count occurrences of each element in both lists
    counter1 = Counter(list1)
    counter2 = Counter(list2)

    # Adjust for common elements
    for element in set(list1) & set(list2):
        common_count = min(counter1[element], counter2[element])
        counter1[element] -= common_count
        counter2[element] -= common_count

    # Reconstruct lists without common elements
    unique_list1 = [item for item in counter1.elements()]
    unique_list2 = [item for item in counter2.elements()]

    # Calculate means, avoiding division by zero
    mean_list1 = sum(unique_list1) / len(unique_list1) if unique_list1 else 0
    mean_list2 = sum(unique_list2) / len(unique_list2) if unique_list2 else 0

    # Calculate and return the difference of means
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

    while True:
        A_changed = False
        B_changed = False

        matching, current_cost, cost_matrix = find_maximum_matching_v3(A, B, merge_bonus, merge_windows)
        if debug:
            print("Current cost of the solution:", current_cost)
            # print_matching(matching, A, B, cost_matrix)
    
        if len(A) >= 2:
            merge_list = []
            for i in range(len(A) - 1):
                new_A = deepcopy(A)
                new_A[i] = merge_blocks_wo_check(new_A[i], new_A[i + 1])
                new_A.pop(i + 1)
    
                updated_matching, updated_cost, cost_matrix = find_maximum_matching_v3(new_A, B, merge_bonus, merge_windows)
                diff = difference_of_means(current_cost, updated_cost)
                if  diff > 0.05:
                    merge_list.append([i, i + 1, diff])
                    if debug:
                        print(new_A[i]['text'], diff)

            merge_list.sort(key=sortFn, reverse=True)
            if len(merge_list) > 0:
                A_changed = True
                A = merge_blocks_by_list(A, merge_list)
                matching, current_cost, cost_matrix = find_maximum_matching_v3(A, B, merge_bonus, merge_windows)
                if debug:
                    print("Cost after optimization A:", current_cost)

        if len(B) >= 2:
            merge_list = []
            for i in range(len(B) - 1):
                new_B = deepcopy(B)
                new_B[i] = merge_blocks_wo_check(new_B[i], new_B[i + 1])
                new_B.pop(i + 1)
    
                updated_matching, updated_cost, cost_matrix = find_maximum_matching_v3(A, new_B, merge_bonus, merge_windows)
                diff = difference_of_means(current_cost, updated_cost)
                if diff > 0.05:
                    merge_list.append([i, i + 1, diff])
                    if debug:
                        print(new_B[i]['text'], diff)

                """
                if "bpk-s integration kam-titan" in new_B[i]['text']:
                    print(updated_cost)
                # """
            
            merge_list.sort(key=sortFn, reverse=True)
            if len(merge_list) > 0:
                B_changed = True
                B = merge_blocks_by_list(B, merge_list)
                matching, current_cost, cost_matrix = find_maximum_matching_v3(A, B, merge_bonus, merge_windows)
                if debug:
                    print("Cost after optimization B:", current_cost)

        if not A_changed and not B_changed:
            break
    matching, _, _ = find_maximum_matching_v3(A, B, consecutive_bonus, window_size)
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


def mask_bounding_boxes(image, bounding_boxes):
    width, height = image.size
    draw = ImageDraw.Draw(image)

    for bbox in bounding_boxes:
        x_ratio, y_ratio, w_ratio, h_ratio = bbox
        x = x_ratio * width
        y = y_ratio * height
        w = w_ratio * width
        h = h_ratio * height
        draw.rectangle([x, y, x + w, y + h], fill="white")

    return image


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


def visual_eval_v3(gpt_img, original_img, print_all=False, ocr_free=True, debug=False):
    """
    gpt_img: file to image rendered by gpt gen code. Please place the html file with the same name in the same folder.
    original_img: file to image rendered by the original code. Please place the html file with the same name in the same folder.
    print_all: print matched information or not. Default to False.
    ocr_free: using ocr free approach or not. Default to True.
    """

    if ocr_free:
        gpt_html = gpt_img.replace(".png", ".html")
        original_html = original_img.replace(".png", ".html")
        os.system(f"python3 {Path(__file__).parent}/screenshot_single.py --html {gpt_html} --png {gpt_img}")
        os.system(f"python3 {Path(__file__).parent}/screenshot_single.py --html {original_html} --png {original_img}")

        blocks1 = get_blocks_ocr_free(gpt_img)
        blocks2 = get_blocks_ocr_free(original_img)
        consecutive_bonus, window_size = 0.1, 1
    else:
        blocks1 = get_ocr_blocks(gpt_img)
        blocks2 = get_ocr_blocks(original_img)
        consecutive_bonus, window_size = 0.25, 2

        blocks1 = merge_blocks(blocks1)
        blocks2 = merge_blocks(blocks2)

    if len(blocks1) == 0 or len(blocks2) == 0:
        return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0)

    blocks1 = merge_blocks_by_bbox(blocks1)
    blocks2 = merge_blocks_by_bbox(blocks2)
    # matching, cost, _ = find_maximum_matching_v3(blocks1, blocks2, consecutive_bonus, window_size)
    blocks1, blocks2, matching = find_possible_merge(blocks1, blocks2, consecutive_bonus, window_size, debug=debug)
    indices1 = [item[0] for item in matching]
    indices2 = [item[1] for item in matching]

    matched_list = []
    scores = []
    max_areas = []
    matched_areas = []
    matched_text_scores = []
    position_scores = []
    text_color_scores = []

    unmatched_area_1 = 0.0
    for i in range(len(blocks1)):
        if i not in indices1:
            unmatched_area_1 += blocks1[i]['bbox'][2] * blocks1[i]['bbox'][3]
    unmatched_area_2 = 0.0
    for j in range(len(blocks2)):
        if j not in indices2:
            unmatched_area_2 += blocks2[j]['bbox'][2] * blocks2[j]['bbox'][3]
    max_areas.append(max(unmatched_area_1, unmatched_area_2))

    for i, j in matching:
        min_block_area = min(blocks1[i]['bbox'][2] * blocks1[i]['bbox'][3], blocks2[j]['bbox'][2] * blocks2[j]['bbox'][3])
        max_block_area = max(blocks1[i]['bbox'][2] * blocks1[i]['bbox'][3], blocks2[j]['bbox'][2] * blocks2[j]['bbox'][3])
        text_similarity = SequenceMatcher(None, blocks1[i]['text'], blocks2[j]['text']).ratio()
        if text_similarity < 0.5:
            max_areas.append(max_block_area)
            continue
        position_similarity = 1 - calculate_distance(blocks1[i]['bbox'][0] + blocks1[i]['bbox'][2] / 2, \
                                                blocks1[i]['bbox'][1] + blocks1[i]['bbox'][3] / 2, \
                                                blocks2[j]['bbox'][0] + blocks2[j]['bbox'][2] / 2, \
                                                blocks2[j]['bbox'][1] + blocks2[j]['bbox'][3] / 2) / np.sqrt(2)
        # scale to 0.5 ~ 1.0
        text_color_similarity = color_similarity(blocks1[i]['color'], blocks2[j]['color']) * 0.5 + 0.5
        matched_list.append([blocks1[i]['bbox'], blocks2[j]['bbox']])

        # validation check
        if min(blocks1[i]['bbox'][2], blocks2[j]['bbox'][2], blocks1[i]['bbox'][3], blocks2[j]['bbox'][3]) == 0:
            print(f"{blocks1[i]} matched with {blocks2[j]}")
        assert calculate_ratio(blocks1[i]['bbox'][2], blocks2[j]['bbox'][2]) > 0 and calculate_ratio(blocks1[i]['bbox'][3], blocks2[j]['bbox'][3]) > 0, f"{blocks1[i]} matched with {blocks2[j]}"

        scores.append(max_block_area * text_similarity * position_similarity * text_color_similarity)
        matched_areas.append(max_block_area)
        max_areas.append(max_block_area)
        matched_text_scores.append(text_similarity)
        position_scores.append(position_similarity)
        text_color_scores.append(text_color_similarity)

        if debug:
            print(f"{blocks1[i]} matched with {blocks2[j]}")
            print(SequenceMatcher(None, blocks1[i]['text'], blocks2[j]['text']).ratio())
            print("text similarity score", text_similarity)
            print("position score", position_similarity)
            print("color score", text_color_similarity)
            print("----------------------------------")
            pass
    
    if print_all:
        print(f"Matched: {len(location_score)}")
        print("Score:")
        print_stat(scores)

    if debug:
        img1 = cv2.imread(gpt_img)
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

    if len(scores) > 0:
        matched = len(scores)

        final_size_score = np.sum(matched_areas) / np.sum(max_areas)
        final_matched_text_score = np.mean(matched_text_scores)
        final_position_score = np.mean(position_scores)
        final_text_color_score = np.mean(text_color_scores)
        final_clip_score =  calculate_clip_similarity_with_blocks(gpt_img, original_img, blocks1, blocks2)
        final_score = np.sum(scores) / np.sum(max_areas) * final_clip_score
        return matched, final_score, (final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score)
    else:
        return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0)


def truncate_repeated_html_elements(soup, max_count=50):
    """
    Truncate HTML elements whose exact HTML content appears more than `max_count` times.

    Args:
    html (str): A string containing HTML content.
    max_count (int): The maximum allowed occurrences for identical content.

    Returns:
    str: The modified HTML with repeated elements truncated.
    """
    # Dictionary to keep track of exact HTML content occurrences
    content_counts = {}

    # Iterate over all elements
    for element in soup.find_all(True):
        # Skip non-tag elements like NavigableString or Comment
        if isinstance(element, (NavigableString, Comment)):
            continue
        
        try:
            element_html = str(element)
        except:
            element.decompose()
            continue
        content_counts[element_html] = content_counts.get(element_html, 0) + 1

        # Check if the element's exact HTML content exceeds max_count
        if content_counts[element_html] > max_count:
            # Remove or truncate the element
            element.decompose()  # or element.string = '...' to truncate

    # Return the modified HTML
    return str(soup)


def make_html(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Use regular expression to check for a pattern that starts with <html (ignoring any attributes)
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
    # print(input_list)
    predict_html_list, original_html = input_list[0], input_list[1]
    predict_img_list = [html.replace(".html", ".png") for html in predict_html_list]
    try:
        predict_blocks_list = []
        for predict_html in predict_html_list:
            # This will help fix some html syntax error
            # predict_html = predict_img.replace(".png", ".html")
            predict_img = predict_html.replace(".html", ".png")
            pre_process(predict_html)
            os.system(f"python3 screenshot_single.py --html {predict_html} --png {predict_img}")
            predict_blocks = get_blocks_ocr_free(predict_img)
            predict_blocks_list.append(predict_blocks)

        # original_html = original_img.replace(".png", ".html")
        original_img = original_html.replace(".html", ".png")
        os.system(f"python3 screenshot_single.py --html {original_html} --png {original_img}")
        original_blocks = get_blocks_ocr_free(original_img)
        original_blocks = merge_blocks_by_bbox(original_blocks)

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
            max_areas = []
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
            max_areas.append(max(unmatched_area_1, unmatched_area_2))
        
            for i, j, text_similarity in matching:
                max_block_area = max(predict_blocks_m[i]['bbox'][2] * predict_blocks_m[i]['bbox'][3], original_blocks_m[j]['bbox'][2] * original_blocks_m[j]['bbox'][3])

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
        
                max_areas.append(max_block_area)
                matched_areas.append(max_block_area)
                # v1: weighted average
                # matched_text_scores.append(max_block_area * text_similarity)
                # position_scores.append(max_block_area * position_similarity)
                # text_color_scores.append(max_block_area * text_color_similarity)
                # v2: average
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
                sum_max_areas = np.sum(max_areas)
        
                final_size_score = np.sum(matched_areas) / np.sum(max_areas)
                # v1: weighted average
                # final_matched_text_score = np.sum(matched_text_scores) / np.sum(max_areas)
                # final_position_score = np.sum(position_scores) / np.sum(max_areas)
                # final_text_color_score = np.sum(text_color_scores) / np.sum(max_areas)

                # v2: average
                final_matched_text_score = np.mean(matched_text_scores)
                final_position_score = np.mean(position_scores)
                final_text_color_score = np.mean(text_color_scores)
                final_clip_score = calculate_clip_similarity_with_blocks(predict_img_list[k], original_img, predict_blocks, original_blocks)

                # final_score = 0.25 * (3 - (3 - final_matched_text_score - final_position_score - final_text_color_score) * np.sum(max_areas) + final_clip_score)
                # v1: weighted average
                # final_score = 0.25 * (final_matched_text_score + final_position_score + final_text_color_score + final_clip_score)
                # v2: average
                final_score = 0.2 * (final_size_score + final_matched_text_score + final_position_score + final_text_color_score + final_clip_score)
                return_score_list.append([sum_max_areas, final_score, (final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score)])
            else:
                print("[Warning] No matched blocks in: ", predict_img_list[k])
                final_clip_score = calculate_clip_similarity_with_blocks(predict_img_list[k], original_img, predict_blocks, original_blocks)
                return_score_list.append([0.0, 0.2 * final_clip_score, (0.0, 0.0, 0.0, 0.0, final_clip_score)])
        return return_score_list
    except:
        print("[Warning] Error not handled in: ", input_list)
        return [[0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0)] for _ in range(len(predict_html_list))]


if __name__ == "__main__":
    reference_dir = "../../testset_100"
    predictions_dir = "../../predictions_100/gpt4v"
    all_scores = 0
    counter = 0

    for filename in tqdm([item for item in os.listdir(predictions_dir) if item.endswith(".html")]):
        if filename.endswith(".html"):
            # matched, loc_score, size_score, final_score = visual_eval(os.path.join(predictions_dir, filename.replace(".html", ".png")), os.path.join(reference_dir, filename.replace(".html", ".png")))
            matched, final_score = visual_eval_v1(os.path.join(predictions_dir, filename.replace(".html", ".png")), os.path.join(reference_dir, filename.replace(".html", ".png")))
            print (filename, matched, final_score)
            all_scores += final_score
            counter += 1
    
    print ("\n")
    print ("avg score: ", all_scores / counter)
    
    