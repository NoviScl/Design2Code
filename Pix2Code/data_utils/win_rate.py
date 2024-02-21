import csv
import os
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np


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


# Open the CSV file
with open('/Users/zhangyanzhe/Downloads/3_1.csv', 'r') as file:
    reader = csv.reader(file)

    img_folder = "/Users/zhangyanzhe/Downloads/sampled_for_annotation_v2"
    baseline = "gpt4v_direct_prompting"
    # tested = "gpt4v_visual_revision_prompting"
    tested = "gemini_direct_prompting"
    baseline_win = 0
    tested_win = 0
    tie = 0

    # Transpose rows to columns
    columns = zip(*reader)
    res = []

    # Iterate over columns
    for column in columns:
        column = list(column)
        # Process each column (each 'column' is a tuple of the column's values)
        if "Overall" in column[0] or "generally speaking" in column[0]:

            res.append(column[1:])
            num = int(column[0].split(" ")[1])

            win1 = column.count('Example 1 better')
            win2 = column.count('Example 2 better')
            tie12 = column.count('Tie')

            if os.path.isfile(os.path.join(img_folder, "testset_full_" + baseline + "_" + tested + "_" + str(num) + ".png")):
                if win1 >= 2:
                    baseline_win += 1
                elif win2 >= 2:
                    tested_win += 1
                else:
                    tie += 1
            elif os.path.isfile(os.path.join(img_folder, "testset_full_" + tested + "_" + baseline + "_" + str(num) + ".png")):
                if win2 >= 2:
                    baseline_win += 1
                elif win1 >= 2:
                    tested_win += 1
                else:
                    tie += 1
            else:
                print(num)
                raise NotImplementedError
            print(baseline_win, tie, tested_win)
    # print(res)
    # Calculate Krippendorff's Alpha for the example data
    alpha = calculate_krippendorff_alpha(res)
    print("Krippendorff's Alpha:", alpha)
    kappa = calculate_fleiss_kappa(res)
    print("Fleiss' Kappa:", kappa)
    print(baseline_win, tie, tested_win)