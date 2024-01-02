import os
import json 

with open("visual_dups.json", "r") as f:
    visual_dups = json.load(f)
print ("#visual dups:", len(visual_dups))

dir = "testset_manual_filtered"
for dup in visual_dups:
    html = os.path.join(dir, str(dup) + ".html")
    png = os.path.join(dir, str(dup) + ".png")
    if os.path.exists(html):
        os.remove(html)
        os.remove(png)

file_count = len([entry for entry in os.listdir(dir) if os.path.isfile(os.path.join(dir, entry)) and ".png" in entry])
print ("#files:", file_count)

