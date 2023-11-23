import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import math
import random
import os
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

import pytesseract
from PIL import Image, ImageDraw

pytesseract.pytesseract.tesseract_cmd = '/sailhome/clsi/bin/tesseract'

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
    # print(cost_matrix[8, 8], cost_matrix[8, 26], cost_matrix[13, 8], cost_matrix[13, 26])
    # print(cost_matrix[8 - 2:8 + 3, 8 - 2:8 + 3], "\n\n", cost_matrix[8 - 2:8 + 3, 26 - 2: 26 + 3], "\n\n", cost_matrix[13 - 2: 13 + 3, 8 - 2:8 + 3], "\n\n", cost_matrix[13 - 2: 13 + 3, 26 - 2: 26 + 3])
    cost_matrix = adjust_cost_for_context(cost_matrix, consecutive_bonus, window_size)
    # print(cost_matrix[8, 8], cost_matrix[8, 26], cost_matrix[13, 8], cost_matrix[13, 26])
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

def calculate_ratio(h1, h2):
    return max(h1, h2) / min(h1, h2)

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def print_stat(alist):
    if len(alist) == 0:
        print("Empty list!")
        return
    # print("Mean:", np.mean(alist),"Median:", np.median(alist),"Min:", min(alist),"Max:", max(alist))
    return np.mean(alist), np.median(alist), min(alist), max(alist)

def print_stat_geo(alist):
    if len(alist) == 0:
        print("Empty list!")
        return
    # print("Geo Mean:", geo_mean(alist),"Median:", np.median(alist),"Min:", min(alist),"Max:", max(alist))
    return geo_mean(alist), np.median(alist), min(alist), max(alist)

def visual_eval(gpt_img, original_img, print_all=False):
    blocks1 = get_ocr_blocks(gpt_img)
    blocks2 = get_ocr_blocks(original_img)

    blocks1 = merge_blocks(blocks1)
    blocks2 = merge_blocks(blocks2)

    matching = find_maximum_matching(blocks1, blocks2, 0.25, 2)
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
        assert calculate_ratio(blocks1[i]['bbox'][2], blocks2[j]['bbox'][2]) > 0 and calculate_ratio(blocks1[i]['bbox'][3], blocks2[j]['bbox'][3]) > 0, f"{blocks1[i]} matched with {blocks2[j]}"
        size_score.append(calculate_ratio(blocks1[i]['bbox'][2], blocks2[j]['bbox'][2]) * calculate_ratio(blocks1[i]['bbox'][3], blocks2[j]['bbox'][3]))
    if print_all:
        print(f"Matched: {len(location_score)}")
        print("Location Score:")
        print_stat(location_score)
        print("Size Score:")
        print_stat_geo(size_score)
    

    img1 = cv2.imread(gpt_img)
    img2 = cv2.imread(original_img)

    img1_with_boxes, img2_with_boxes = draw_matched_bboxes(img1, img2, matched_list)

    cv2.imwrite(gpt_img.replace(".png", "_demo.png"), img1_with_boxes)
    cv2.imwrite(original_img.replace(".png", "_demo.png"), img2_with_boxes)

    if len(location_score) > 0:
        matched = len(location_score)
        loc_score = np.mean(location_score)
        s_score = geo_mean(size_score)
        final_score = loc_score / np.sqrt(2) + (1 - 1 / s_score)
        return matched, loc_score, s_score, final_score
    else:
        return 0.0, None, None, None

if __name__ == "__main__":
    # check_match_images('trial_dataset/rick.jpg', 'syn_dataset/diyi_gpt4.png', True)
   matched, loc_score, size_score, final_score = visual_eval('syn_dataset/diyi_gpt4.png', 'syn_dataset/diyi.png')
   print (matched, loc_score, size_score, final_score)
