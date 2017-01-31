from requests import get
from bs4 import BeautifulSoup
from unidecode import unidecode
import pdb

def get_content(url):
    response = get(url)
    html = BeautifulSoup(response.content, 'html.parser')
    return html

def get_headers(html):
    tab = html.findAll('div', attrs = {'class': 'innerContainer'})
    # table = html.findAll('span', {'class': 'blist-th-name'})
    return tab

if __name__ == '__main__':
    aco = get_content('https://data.cms.gov/ACO/Medicare-Shared-Savings-Program-Accountable-Care-O/ucce-hhpu')
    # headers = get_headers(aco)

    f = open('html_content.txt', 'w')
    f.write(unidecode(aco.get_text()))
    f.close()
