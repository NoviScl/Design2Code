import csv
import os
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
from copy import deepcopy

def calculate_fleiss_kappa(data):
    # Convert text responses to a count matrix
    # "Example 1 better" is column 0, "Example 2 better" is column 1, "Tie" is column 2
    count_matrix = np.zeros((len(data), 2))
    
    for i, item in enumerate(data):
        count_matrix[i, 0] = item.count('Left')
        count_matrix[i, 1] = item.count('Right')

    # Calculate Fleiss' Kappa
    kappa = fleiss_kappa(count_matrix, method='fleiss')
    
    return kappa


def get_res(columns, j):
    baseline_win = 0
    tested_win = 0
    res = []

    # Iterate over columns
    for column in deepcopy(columns):
        column = list(column)

        # Process each column (each 'column' is a tuple of the column's values)
        if "Question" in column[0]:

            num = int(column[0].split(" ")[1])
            if num == 0:
                continue
                
            column = column[:j] + column[j+1:]
            res.append(column[1:])

            win1 = column.count('Left')
            win2 = column.count('Right')

            assert win1 + win2 == 5

            if win1 >= 3:
                if os.path.exists(f"/Users/zhangyanzhe/Downloads/singlestudy/testset_full_gpt4v_visual_revision_prompting_v2_{num}.png"):
                    baseline_win += 1
                elif os.path.exists(f"/Users/zhangyanzhe/Downloads/singlestudy/gpt4v_visual_revision_testset_full_prompting_v2_{num}.png"):
                    tested_win += 1
            else:
                if os.path.exists(f"/Users/zhangyanzhe/Downloads/singlestudy/testset_full_gpt4v_visual_revision_prompting_v2_{num}.png"):
                    tested_win += 1
                elif os.path.exists(f"/Users/zhangyanzhe/Downloads/singlestudy/gpt4v_visual_revision_testset_full_prompting_v2_{num}.png"):
                    baseline_win += 1
        print(baseline_win, tested_win)
    return baseline_win, tested_win, res

# Open the CSV file
with open(f'/Users/zhangyanzhe/Downloads/singlev32.csv', 'r') as file:
    reader = csv.reader(file)

    # Transpose rows to columns
    columns = zip(*reader)

    kappa_list = []

    for j in range(1, 7):
        baseline_win, tested_win, res = get_res(columns, j)
        kappa = calculate_fleiss_kappa(res)
        print("Fleiss' Kappa:", kappa)
        print(baseline_win, tested_win)
        kappa_list.append(kappa)
    print(kappa_list)
    baseline_win, tested_win, res = get_res(columns, 1 + kappa_list.index(max(kappa_list)))
    kappa = calculate_fleiss_kappa(res)
    print("Fleiss' Kappa:", kappa)
    print(baseline_win, tested_win)