import os 
import re 
import shutil

def replace_filename_with_rick(input_string):
    # Use regular expression to replace any filename ending with .jpg inside quotations with "rick.jpg"
    return re.sub(r'"([^"]*\.jpg)"', r'"rick.jpg"', input_string)

if __name__ == "__main__":
    input_dirs = ["../../websight_predictions_full", "../../pix2code_predictions_full"]
    for input_dir in input_dirs:
        os.makedirs(input_dir + "_with_placeholder", exist_ok=True)
        for filename in os.listdir(input_dir):
            if filename.endswith(".html"):
                with open(os.path.join(input_dir, filename), "r") as f:
                    content = f.read()
                    content = replace_filename_with_rick(content)
                with open(os.path.join(input_dir + "_with_placeholder", filename), "w") as f:
                    f.write(content)
            elif filename == "rick.jpg":
                shutil.copy(os.path.join(input_dir, filename), os.path.join(input_dir + "_with_placeholder", filename))
