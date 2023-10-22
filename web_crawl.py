import requests
from bs4 import BeautifulSoup

def get_page(url, output_file="diyi.html"):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    html = soup.prettify()
    with open(output_file, "w") as f:
        f.write(html)

get_page(url = "https://cs.stanford.edu/~diyiy/research.html", output_file="diyi.html")

