import requests
import bs4
import nltk
import re

"""URLS AND DATA SCRAPING"""

data = [
    'https://sl.wikipedia.org/wiki/Domači_pes',
    'https://sl.wikipedia.org/wiki/Nebinovke',
    'https://sl.wikipedia.org/wiki/Natrij',
    'https://sl.wikipedia.org/wiki/Kefren',
    'https://sl.wikipedia.org/wiki/Kristalna_noč'
]

html = requests.get(data[0])
data = bs4.BeautifulSoup(html.text, "lxml")

title = data.find(name='h1', id='firstHeading', class_='firstHeading')
print(title.text)

all_text = data.find(name='div', class_='mw-parser-output')

first_child = None
for i in all_text.children:
    first_child = i
    break

tag = None  # first paragraph of summary

if first_child.name != 'p':
    tag = first_child.findNextSibling(name='p')
else:
    tag = all_text.find('p')

summary = [tag.text]
after = None

while True:  # next paragraphs of summary until div tag
    tag = tag.nextSibling
    if isinstance(tag, bs4.element.Tag):
        if tag.name == 'h2':
            after = tag
            break
        elif tag.name == 'p':
            summary.append(tag.text)

tags = after.findNextSiblings(name='p')  # all paragraphs of text
text = []
for t in tags:
    text.append(t.text)

#print(summary)
#print(text)

"""CLEANING AND TOKENIZING DATA"""

summary_combined = ''
for paragraph in summary:
    summary_combined = summary_combined + paragraph

text_combined = ''
for paragraph in text:
    text_combined = text_combined + paragraph

summary_cleaned = re.sub('\[[0-9]*\]', '', summary_combined)
text_cleaned = re.sub('\[[0-9]*\]', '', text_combined)
#print(summary_combined)
#print(summary_cleaned)
#print(text_combined)
#print(text_cleaned)

summary_tokens = nltk.word_tokenize(summary_cleaned)
text_tokens = nltk.word_tokenize(text_cleaned)

# exclude (,.

exclude_list = [',', '.', '(', ')', '»', '«']
processed_summary_tokens = [e.lower() for e in summary_tokens if e not in exclude_list]
processed_text_tokens = [e.lower() for e in text_tokens if e not in exclude_list]
all_tokens = processed_summary_tokens + processed_text_tokens

print(processed_summary_tokens)
print(processed_text_tokens)

"""TEXT DATA"""

freq_distribution = nltk.FreqDist(all_tokens)
#print(freq_distribution)
#print(freq_distribution.most_common(10))

nltk_text = nltk.Text(all_tokens)
#print(nltk_text.collocations())
