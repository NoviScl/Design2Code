import difflib
import os

def check_repetitive_content(file_path, chunk_size=100, repetition_threshold=3, similarity_threshold=0.9):
    """
    Checks for repetitive content in a text file, considering both exact and similar chunks.

    :param file_path: Path to the text file.
    :param chunk_size: The size of each chunk for comparison.
    :param repetition_threshold: Minimum number of repetitions to consider it as repetitive content.
    :param similarity_threshold: The threshold for considering two chunks as similar (0 to 1).
    :return: A tuple indicating if repetitive content was found and the position where it starts.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split content into chunks
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

    # Check for repetitive and similar chunks
    seen = {}
    repetitive_start = len(content)
    for i, chunk in enumerate(chunks):
        for seen_chunk, indexes in seen.items():
            similarity = difflib.SequenceMatcher(None, chunk, seen_chunk).ratio()
            if similarity >= similarity_threshold:
                indexes.append(i)
                if len(indexes) >= repetition_threshold:
                    repetitive_start = min(repetitive_start, indexes[0] * chunk_size)
                break
        else:
            seen[chunk] = [i]

    repetitive, start_position =  repetitive_start != len(content), repetitive_start

    if repetitive:
        print(f"[Warning] Repetitive content found in {file_path}, start at {start_position}")
        print(f"[Warning] You might want to manually check whether the automatic repetition removal is correct.")
        os.rename(file_path, file_path.replace(".html", "_old.txt"))
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content[:start_position])

check_repetitive_content("/Users/zhangyanzhe/Downloads/dup_1.html")
check_repetitive_content("/Users/zhangyanzhe/Downloads/dup_2.html")
check_repetitive_content("/Users/zhangyanzhe/Downloads/dup_3.html")
check_repetitive_content("/Users/zhangyanzhe/Downloads/dup_4.html")
check_repetitive_content("/Users/zhangyanzhe/Downloads/dup_5.html")
check_repetitive_content("/Users/zhangyanzhe/Downloads/dup_6.html")