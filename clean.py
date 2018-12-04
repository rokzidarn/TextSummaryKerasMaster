import requests
import bs4
import nltk
import re

def get_data(url):
    html = requests.get(url)
    data = bs4.BeautifulSoup(html.text, "lxml")

    title_tag = data.find(name='h1', id='firstHeading', class_='firstHeading')
    title = title_tag.text

    all_text = data.find(name='div', class_='mw-parser-output')
    first_child = None
    for i in all_text.children:
        first_child = i
        break
    if first_child.name != 'p':
        tag = first_child.findNextSibling(name='p')
    else:
        tag = all_text.find('p')

    summary = [tag.text]

    while True:  # next paragraphs of summary until div tag
        tag = tag.nextSibling
        if isinstance(tag, bs4.element.Tag):
            if tag.name == 'h2':
                after = tag
                break
            elif tag.name == 'p':
                summary.append(tag.text)

    tags = after.findNextSiblings(name='p')  # all paragraphs of article
    article = []
    for t in tags:
        article.append(t.text)

    summary_combined = ''
    for paragraph in summary:
        summary_combined = summary_combined + paragraph

    text_combined = ''
    for paragraph in article:
        text_combined = text_combined + paragraph

    return (title, summary_combined, text_combined)

def clean_data(text):
    text_cleaned = re.sub('\[[0-9]*\]', '', text)  # remove bibliography links
    tokens = nltk.word_tokenize(text_cleaned)

    exclude_list = [',', '.', '(', ')', '»', '«']  # keeps dates and numbers, excludes the rest
    processed_tokens = [e.lower() for e in tokens if e not in exclude_list]

    return processed_tokens

# MAIN

data = [
    'https://sl.wikipedia.org/wiki/Domači_pes',
    'https://sl.wikipedia.org/wiki/Nebinovke',
    'https://sl.wikipedia.org/wiki/Natrij',
    'https://sl.wikipedia.org/wiki/Kefren',
    'https://sl.wikipedia.org/wiki/Kristalna_noč'
]

(title, summary, article) = get_data(data[0])
#print(title, '\n', summary, '\n', article, '\n')

summary_clean = clean_data(summary)
article_clean = clean_data(article)

print(summary_clean)
print(len(summary_clean))
print(article_clean)
print(len(article_clean))

freq_distribution = nltk.FreqDist(summary_clean+article_clean)  # all tokens from summary and text
vocabulary_size = len(freq_distribution.items())

print(freq_distribution.most_common(10))
print(vocabulary_size)
