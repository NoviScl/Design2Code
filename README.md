# Pix2Code: Benchmarking Multimodal Code Generation

## Data

- Test set: The full official test set is availble for downloading at [this link](https://drive.google.com/file/d/1AdqgWx8wgz_GM1qeupY1eyUiT7E2zo6_/view?usp=sharing). It contains 484 screenshot-code pairs. 

- GPT-4V and Gemini-Pro predictions: All predictions (including rendered webpage screenshots) are available for downloading at [this link](https://drive.google.com/file/d/1zinGz87_4Y-YIkeA4uPgaoxITAMjjObH/view?usp=sharing).

- Websight predictions: [this link](https://drive.google.com/file/d/1pNmAiGC259t_1VBfNeq7JI98RSM7zMHo/view?usp=sharing)

- Pix2Code-18B predictions: [this link](https://drive.google.com/file/d/16meY5D_TWiXo7K1IUMLjoXhFH6DaSbLO/view?usp=sharing)

- Sampled predictions for human evaluation: The sampled predictions used to obtain human evaluation can be found [here](https://drive.google.com/file/d/1L3tj35o9QiWEcDH95XpGFAZUij6LNAHu/view?usp=sharing).

![](example.png)


## Code Structure

- `data_utils` contains the filtering and processing scripts to construct the test data from C4. 

- `metrics` contains the metric scripts for the evaluation.

- `prompting` contains the code for running all the prompting experiments, including the actual prompts used. 
