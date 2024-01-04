import os

def remove_import_lines(folder_path):
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an HTML file
        if filename.endswith(".html"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Remove lines that start with "@import"
            new_lines = [line for line in lines if not line.strip().startswith("@import")]

            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.writelines(new_lines)

# Usage example
folder_path = './pilot_testset/'  # Replace with your folder path
remove_import_lines(folder_path)