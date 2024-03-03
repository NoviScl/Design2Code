from data_stats import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr



def line_plot():
    method_name = "gpt4v_visual_revision_prompting"

    ## load the test scores of GPT-4V 
    with open("../metrics/prediction_file_name_list_part1_new.json", "r") as f:
        file_name_list = json.load(f)
    with open("../metrics/res_dict_part1_new.json", "r") as f:
        res_dict = json.load(f)
    res_lst = res_dict[method_name]

    x_variables = []
    for filename in file_name_list:
        with open("../../testset_final/" + filename, "r") as f:
            html_content = f.read() 
            variable = count_total_nodes(html_content)
            x_variables.append(variable)

    y_variables = []
    for i, res in enumerate(res_lst):
        y_variables.append({"performance": res[1], "tags": x_variables[i], "metric": "block"})
        # y_variables.append({"performance": res[2], "tags": x_variables[i], "metric": "text"})
        y_variables.append({"performance": res[3], "tags": x_variables[i], "metric": "position"})
        # y_variables.append({"performance": res[4], "tags": x_variables[i], "metric": "color"})
        y_variables.append({"performance": res[5], "tags": x_variables[i], "metric": "CLIP"})

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(y_variables)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='tags', y='performance', hue='metric', palette='tab10')

    plt.title('Curves for Different Categories')
    plt.xlabel('Variable')
    plt.ylabel('Value')
    plt.legend(title='Category')
    plt.grid(True)
    plt.show()



def bar_plot():
    method_name = "gpt4v_visual_revision_prompting"

    ## load the test scores of GPT-4V 
    with open("../metrics/prediction_file_name_list_part1_new.json", "r") as f:
        file_name_list = json.load(f)
    with open("../metrics/res_dict_part1_new.json", "r") as f:
        res_dict = json.load(f)
    res_lst = res_dict[method_name]

    x_variables = []
    for filename in file_name_list:
        with open("../../testset_final/" + filename, "r") as f:
            html_content = f.read() 
            variable = count_total_nodes(html_content)
            if variable < 141:
                variable = 1 
            elif variable < 270:
                variable = 2
            elif variable < 399:
                variable = 3
            elif variable <= 528:
                variable = 4
            x_variables.append(variable)
    
    y_variables = []
    for i, res in enumerate(res_lst):
        y_variables.append({"performance": res[1], "tags": x_variables[i], "metric": "block"})
        # y_variables.append({"performance": res[2], "tags": x_variables[i], "metric": "text"})
        y_variables.append({"performance": res[3], "tags": x_variables[i], "metric": "position"})
        # y_variables.append({"performance": res[4], "tags": x_variables[i], "metric": "color"})
        y_variables.append({"performance": res[5], "tags": x_variables[i], "metric": "CLIP"})

    # Convert to DataFrame
    df = pd.DataFrame(y_variables)

    # Plotting
    g = sns.catplot(
        data=df, kind="bar",
        x="tags", y="performance", hue="metric",
        ci="sd", palette="dark", alpha=.6, height=6,
        aspect=2
    )
    g.despine(left=True)
    g.set_axis_labels("Tags", "Performance")
    g.legend.set_title("Metrics")
    plt.show()


def correlation():
    method_name = "gpt4v_visual_revision_prompting"

    ## load the test scores of GPT-4V 
    with open("../metrics/prediction_file_name_list_part1_new.json", "r") as f:
        file_name_list = json.load(f)
    with open("../metrics/res_dict_part1_new.json", "r") as f:
        res_dict = json.load(f)
    res_lst = res_dict[method_name]

    x_variables = []
    for filename in file_name_list:
        with open("../../testset_final/" + filename, "r") as f:
            html_content = f.read() 
            variable = count_total_nodes(html_content)
            # variable = count_unique_tags(html_content)
            # variable = calculate_dom_depth(html_content)
            x_variables.append(variable)
    
    y_variables = {"block": [], "text": [], "position": [], "color": [], "CLIP": []}
    for i, res in enumerate(res_lst):
        y_variables["block"].append(res[1])
        y_variables["text"].append(res[2])
        y_variables["position"].append(res[3])
        y_variables["color"].append(res[4])
        y_variables["CLIP"].append(res[5])
    
    for key in y_variables:
        print (key)
        correlation, p = pearsonr(x_variables, y_variables[key])
        print (correlation, p)



def find_prompting_difference():

    ## load the test scores of GPT-4V 
    with open("../metrics/prediction_file_name_list_part1_new.json", "r") as f:
        file_name_list = json.load(f)
    with open("../metrics/res_dict_part1_new.json", "r") as f:
        res_dict = json.load(f)
    direct_prompting = res_dict["gpt4v_direct_prompting"]
    text_augmented_prompting = res_dict["gpt4v_text_augmented_prompting"]
    visual_revision_prompting = res_dict["gpt4v_visual_revision_prompting"]

    differences = []
    for i in range(len(file_name_list)):
        diff = text_augmented_prompting[i][1] - direct_prompting[i][1]
        differences.append(diff)
    
    ## get sorted indices
    sorted_idx = sorted(range(len(differences)), key=lambda k: differences[k])[::-1]
    for idx in sorted_idx[ : 10]:
        print (file_name_list[idx], differences[idx])

    return 

if __name__ == "__main__":
    # correlation()
    find_prompting_difference()
