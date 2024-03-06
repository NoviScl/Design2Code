# Automatic Metrics

Given a reference webpage screenshot $I_R$ and a generated webpage screenshot $I_G$, we use a text detection module to output a set of detected visual element blocks for each: $R = \{r_1, r_2,..., r_m\}$ and $G = \{g_1, g_2,..., g_n\}$, where each block contains its textual content and bounding box coordinates. 

The common approach to detect the texts in a given screenshot is to use OCR tools, which return a list of text segments with their bounding boxes. However, in our case, we find that open-source OCR tools usually output noisy outputs, which may affect the stability of downstream evaluation. Since we already have the source HTML codes for reference webpage screenshots, we apply an alternative approach: we alter the color differently for different text segments in the source HTML code and detect text segments in the webpage by taking two extra screenshots and tracking pixels with different colors (implemented in [ocr_free_utils.py](ocr_free_utils.py)). This helps us locate text segments from the HTML source code in the screenshots without text recognition errors.

In practice, we also use heuristics to detect and delete repetitive generation results (which is an issue mainly for open-source models) using the *check_repetitive_content* function in [dedup_post_gen.py](../data_utils/dedup_post_gen.py) before running evaluation.

Based on the two sets of detected blocks, we use the Jonker-Volgenant algorithm (*scipy.optimize.linear_sum_assignment*) to get the optimal matching $M$ between $R$ and $G$, where $(p, q) \in M$ indicates $r_p$ is matched with $g_q$.

Specifically, we use the negative sequence similarity between textual contents $-\mathbf{sim_{text}}(,)$ to initialize the cost matrix and ignore the matched pairs with a sequence similarity lower than $0.5$. Since detected text blocks might be in different granularity, we also enumerate merging neighbor text blocks to search for matching with the highest similarity (the *find_possible_merge* function in [visual_score.py](visual_score.py)). However, the matching may still not be perfect, especially when there are large granularity differences (our search does not consider merging non-contiguous blocks).

Given $R$, $G$, and matched pairs in $M$, we evaluate similarity along the following aspects (see [visual_score.py](visual_score.py)):

__Block-Match__: 

The first desideratum of the task is that all visual elements from the reference webpage should be reproduced in the generated webpage, and the generated webpage should not hallucinate non-existent new elements. 
We measure this by computing the total sizes of all matched blocks divided by the total sizes of all blocks, including unmatched ones (either because the generated webpages missed them or because the generated webpages contain hallucinated blocks):

$\mathbf{match_{block}}(r_p, g_q) = \frac{ S(r_p) + S(g_q)}{\sum_{(i,j) \in M}  (S(r_i) + S(g_j)) + (\sum_{i \in U_R} S(r_i) + \sum_{j \in U_G} S(g_j))},$
    
$\mathbf{match_{block}}(R, G) = \sum_{(p,q) \in M} \mathbf{match_{block}}(r_p, g_q),$
    
where $S(.)$ returns the size of the blocks, $U_R$ and $U_G$ denotes the unmatched blocks in $R$ and $G$.
    
The intuition here is that unmatched blocks will lower the score as they indicate missing original blocks or generating hallucinated blocks, and the larger the unmatched blocks are, the lower this score is. 
    
__Text__: 

Given two strings from two matched blocks $r_p$ and $g_q$, the text similarity $\mathbf{sim_{text}}(r_p, g_q)$ is calculated as twice the number of overlapping characters divided by the total number of characters in the two strings (character-level Sørensen-Dice similarity). The overall score is averaged across all matched pairs. 

__Position__: 

The positioning of the blocks largely impacts the overall layout. 
For each matched pair $(p, q)$, we calculate the position similarity $\mathbf{sim_{pos}}(r_p, g_q) = 1 - max(abs(x_q - x_p), abs(y_q - y_p))$, where $(x_p, y_p)$ and $(x_q, y_q)$ are normalized coordinates (in $[0, 1]$) of $r_p$ and $g_q$'s centors. The overall score is averaged across all matched pairs.
  
__Color__: 

We use the CIEDE2000 color difference formula to assess the perceptual difference between the colors of the generated text in block $g_q$ and the reference text in block $r_p$, denoted as $\mathbf{sim_{color}}(r_p, g_q))$, where the formula considers the complexities of human color vision. The overall score is averaged across all matched pairs.

__CLIP__:

To evaluate the visual similarity of $I_R$ and $I_G$, we use the similarity of their CLIP embedding, denoted as $\mathbf{CLIP}(I_R, I_G)$. Specifically, we extract features by CLIP-ViT-B/32 after resizing screenshots to squares. To rule out the texts in the screenshots, we use the [OpenCV inpainting](https://docs.opencv.org/4.3.0/df/d3d/tutorial_py_inpainting.html) to mask all detected text boxes using their bounding box coordinates.
