import nltk
import wikipediaapi
import pywikibot
from pywikibot import pagegenerators
import codecs

global_article = []


def get_data(sections, exclusions):
    for s in sections:
        if s.title not in exclusions:
            global_article.append(s.text)
            get_data(s.sections, exclusions)


def write_data(title, dir, text):
    raw_sentences = nltk.sent_tokenize(text)
    sentences = []

    i = 0
    while i < len(list(raw_sentences)):
        s = raw_sentences[i]
        if (s[-2:] == "n." or s[-3:] == "št." or s[-3:] == "oz." or s[-4:] == "npr." or s[-5:] == "angl."
                or s[-4:] == "izg." or s[-3:] == "sv." or s[-4:] == "lat." or s[-3:] == "tj.") and len(sentences) > 0:
            # fix special case sentences
            prev = sentences[-1]
            sentences.pop()
            sentences.append(prev + " " + s + " " + raw_sentences[i + 1])
            i += 1
        else:
            sentences.append(s)
        i += 1

    with codecs.open('data/'+dir+'/'+title+'.txt', 'w', encoding='utf8') as f:  # write to file
        for item in sentences:
            f.write("{}\n".format(item))


# MAIN

wiki = wikipediaapi.Wikipedia('sl')
site = pywikibot.Site()
category_names = ['Category:Naravoslovje', 'Category:Družboslovje', 'Category:Filozofija', 'Category:Geografija',
                  'Category:Ljudje', 'Category:Tehnika', 'Category:Umetnost', 'Category:Zgodovina']
category = pywikibot.Category(site, category_names[0])

generated = pagegenerators.CategorizedPageGenerator(category, recurse=5)
exclusion_sections = ["Glej tudi", "Viri", "Zunanje povezave", "Opombe", "Sklici", "Viri, dodatno branje",
                      "Izbrana bibliografija", "Literatura", "Sklici in opombe", "Zunanje povezave in viri"]

urls, titles, articles, summaries = [], [], [], []
limit = 400

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

    if 800 < article_length < 1600:  # min-max length of article
        limit = limit - 1

        titles.append(title)
        urls.append(url)
        summaries.append(article_summary)
        articles.append(article_text)

        print(title, article_length, url)
        write_data(title, 'summaries', article_summary)
        write_data(title, 'articles', article_text)
