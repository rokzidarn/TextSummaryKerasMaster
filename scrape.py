import nltk
import itertools
import wikipediaapi
import pywikibot
from pywikibot import pagegenerators
import codecs
import os
from pprint import pprint

global_article = []

def get_data(sections, exclusions):
    for s in sections:
        if s.title not in exclusions:
            global_article.append(s.text)
            get_data(s.sections, exclusions)

def clean_data(text):
    tokens = nltk.word_tokenize(text)
    exclude_list = [',', '.', '(', ')', '>', '>', '»', '«', ':', '–', '+', '–', '--',
                    '``', '\"', "\'\'", '?', '!', ';']
    # keeps dates and numbers, excludes the rest
    clean_tokens = [e.lower() for e in tokens if e not in exclude_list]

    return clean_tokens

def write_data(title, dir, text):
    raw_sentences = nltk.sent_tokenize(text)
    sentences = []

    i = 0
    while i < len(list(raw_sentences)):
        s = raw_sentences[i]
        if s[-3:] == "št." or s[-3:] == "oz." or s[-4:] == "npr.":  # fix special case sentences
            prev = sentences[-1]
            sentences.pop()
            sentences.append(prev + " " + s + " " + raw_sentences[i + 1])
            i += 1
        else:
            sentences.append(s)
        i += 1

    #for s in sentences:
        #print(s)

    with codecs.open('data/'+dir+'/'+title+'.txt', 'w', encoding='utf8') as f:  # write to file
        for item in sentences:
            f.write("{}\n".format(item))

def read_data():
    summaries = []
    articles = []

    summary_files = os.listdir("data/summaries/")
    for file in summary_files:
        f = codecs.open("data/summaries/"+file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        summaries.append(' '.join(tmp))

    article_files = os.listdir("data/articles/")
    for file in article_files:
        f = codecs.open("data/articles/"+file, encoding='utf-8')
        tmp = []
        for line in f:
            tmp.append(line)
        articles.append(' '.join(tmp))

    return summaries, articles

# MAIN

wiki = wikipediaapi.Wikipedia('sl')
site = pywikibot.Site()
category_names = ['Category:Naravoslovje', 'Category:Družboslovje', 'Category:Filozofija', 'Category:Geografija',
                  'Category:Ljudje', 'Category:Matematika', 'Category:Tehnika',
                  'Category:Umetnost', 'Category:Zgodovina']
category = pywikibot.Category(site, category_names[6])

generated = pagegenerators.CategorizedPageGenerator(category, recurse=3)
exclusion_sections = ["Glej tudi", "Viri", "Zunanje povezave", "Opombe", "Sklici", "Viri, dodatno branje"]
urls, titles, articles, summaries = [], [], [], []
limit = 3

for page in generated:
    global_article = []
    if limit == 0:
        break

    title = page.title()
    url = page.full_url()

    wiki_page = wiki.page(title)
    article_summary = wiki_page.summary
    get_data(wiki_page.sections, exclusion_sections)
    article_text = ''.join(global_article)
    article_length = len(article_text.split())

    print(".", end="", flush=True)

    if 800 < article_length < 2200:  # min-max length of article
        limit = limit - 1

        titles.append(title)
        urls.append(url)
        summaries.append(article_summary)
        articles.append(article_text)

        print(title, article_length, url)
        #print(article_summary)
        #print(article_text)

        write_data(title, 'summaries', article_summary)
        write_data(title, 'articles', article_text)

summaries_read, articles_read = read_data()
summaries_clean = [clean_data(summary) for summary in summaries_read]
articles_clean = [clean_data(article) for article in articles_read]
#pprint(articles_clean)

all_tokens = list(itertools.chain(*summaries_clean)) + list(itertools.chain(*articles_clean))
freq_distribution = nltk.FreqDist(all_tokens)
vocabulary_size = len(freq_distribution.items())

#print(freq_distribution.most_common(10))
print(vocabulary_size)
