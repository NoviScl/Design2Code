from crawl_w_css import *
from data_clean import *


parser = argparse.ArgumentParser()
parser.add_argument('--partition', type=str, default='00', help='data partition')
args = parser.parse_args()

urls = []
partition = args.partition.strip()
c4 = "/nlp/scr2/nlp/data/commoncrawl/c4/en/c4-train.000{}-of-01024.json".format(str(partition))
with open(c4, 'r') as f:
    for line in f:
        d = json.loads(line)
        urls.append(d["url"])

urls = list(set(urls))
print ("total #urls: ", len(urls))

if not os.path.exists("/juice2/scr2/nlp/pix2code/zyanzhe/c4-train-html-part{}".format(partition)):
    os.makedirs("/juice2/scr2/nlp/pix2code/zyanzhe/c4-train-html-part{}".format(partition))

counter = 1
for i, url in tqdm(enumerate(urls), total=len(urls)):
    html_content = fetch_and_embed_css(url)
    if html_content:
        html_content = all_filters_test(html_content, count_total=False)
        if html_content:
            try:
                with open("/juice2/scr2/nlp/pix2code/zyanzhe/c4-train-html-part{}/{}.html".format(partition, counter), "w") as f:
                    f.write(html_content)
                counter += 1
            except:
                continue