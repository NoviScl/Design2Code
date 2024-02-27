import csv
import os
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
from copy import deepcopy

def calculate_krippendorff_alpha(data):
    # Convert text responses to numerical codes
    # "Example 1 better" is 0, "Example 2 better" is 1, and "Tie" is 2
    numerical_data = [
        [0 if response == 'Example 1 better' else 1 if response == 'Example 2 better' else 2 for response in item] 
        for item in data
    ]

    # Calculate Krippendorff's Alpha
    alpha = krippendorff.alpha(reliability_data=numerical_data, level_of_measurement='nominal')
    
    return alpha


def calculate_fleiss_kappa(data):
    # Convert text responses to a count matrix
    # "Example 1 better" is column 0, "Example 2 better" is column 1, "Tie" is column 2
    count_matrix = np.zeros((len(data), 3))
    
    for i, item in enumerate(data):
        count_matrix[i, 0] = item.count('Example 1 better')
        count_matrix[i, 1] = item.count('Example 2 better')
        count_matrix[i, 2] = item.count('Tie')

    # Calculate Fleiss' Kappa
    kappa = fleiss_kappa(count_matrix, method='fleiss')
    
    return kappa


tested_dict = {
    "1v2": "gpt4v_visual_revision_prompting",
    "2v2": "gpt4v_text_augmented_prompting",
    "3v2": "gpt4v_direct_prompting",
    "4v2": "gemini_text_augmented_prompting",
    "5v2": "gemini_visual_revision_prompting",
    "6v2": "pix2code",
    "7v2": "websight_predictions_full",
}
check_id = "7v2"


def get_res(columns, j):
    baseline_win = 0
    tested_win = 0
    tie = 0
    res = []
    ann = []

    # Iterate over columns
    for column in deepcopy(columns):
        column = list(column)

        # Process each column (each 'column' is a tuple of the column's values)
        if "Overall" in column[0] or "generally speaking" in column[0]:

            num = int(column[0].split(" ")[1])
            if num == 0:
                continue
                
            column = column[:j] + column[j+1:]
            res.append(column[1:])

            win1 = column.count('Example 1 better')
            win2 = column.count('Example 2 better')
            tie12 = column.count('Tie')

            assert win1 + win2 + tie12 == 5

            if os.path.isfile(os.path.join(img_folder, "testset_full_" + baseline + "_" + tested + "_" + str(num) + ".png")):
                if win1 >= 3:
                    baseline_win += 1
                    ann.append("baseline")
                elif win2 >= 3:
                    tested_win += 1
                    ann.append("tested")
                else:
                    tie += 1
                    ann.append("tie")
            elif os.path.isfile(os.path.join(img_folder, "testset_full_" + tested + "_" + baseline + "_" + str(num) + ".png")):
                if win2 >= 3:
                    baseline_win += 1
                    ann.append("baseline")
                elif win1 >= 3:
                    tested_win += 1
                    ann.append("tested")
                else:
                    tie += 1
                    ann.append("tie")
            else:
                print(num)
                raise NotImplementedError
    return baseline_win, tested_win, tie, res, ann

# Open the CSV file
with open(f'/Users/zhangyanzhe/Downloads/{check_id}.csv', 'r') as file:
    reader = csv.reader(file)

    img_folder = "/Users/zhangyanzhe/Downloads/sampled_for_annotation_v4"
    baseline = "gemini_direct_prompting"
    # tested = "gpt4v_visual_revision_prompting"
    tested = tested_dict[check_id]

    # Transpose rows to columns
    columns = zip(*reader)

    kappa_list = []

    for j in range(1, 7):
        baseline_win, tested_win, tie, res, _ = get_res(columns, j)
        kappa = calculate_fleiss_kappa(res)
        print("Fleiss' Kappa:", kappa)
        kappa_list.append(kappa)
    print(kappa_list)
    baseline_win, tested_win, tie, res, ann = get_res(columns, 1 + kappa_list.index(max(kappa_list)))
    kappa = calculate_fleiss_kappa(res)
    print("Fleiss' Kappa:", kappa)
    print(baseline_win, tie, tested_win)

    with open(f"/Users/zhangyanzhe/Downloads/{check_id}.txt", "w") as text_file:
        text_file.write("\n".join(ann))