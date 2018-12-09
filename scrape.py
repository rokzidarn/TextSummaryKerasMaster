import wikipediaapi
import pywikibot
from pywikibot import pagegenerators

global_article = []

def get_article(sections, exclusions):
    for s in sections:
        if s.title not in exclusions:
            global_article.append(s.text)
            get_article(s.sections, exclusions)

# MAIN

wiki = wikipediaapi.Wikipedia('sl')
site = pywikibot.Site()
category = pywikibot.Category(site, 'Category:Naravoslovje')

generated = pagegenerators.CategorizedPageGenerator(category, recurse=2)
urls = []
titles = []
articles = []
summaries = []

exclusion_sections = ["Glej tudi", "Viri", "Zunanje povezave", "Opombe", "Sklici"]

limit = 5

for page in generated:
    global_article = []
    if limit == 0:
        break

    title = page.title()
    url = page.full_url()

    wiki_page = wiki.page(title)
    article_summary = wiki_page.summary
    get_article(wiki_page.sections, exclusion_sections)
    article_text = ''.join(global_article)
    article_length = len(article_text.split())
    print(".", end="", flush=True)

    if 800 < article_length < 2200:
        limit = limit - 1

        titles.append(title)
        urls.append(url)
        summaries.append(article_summary)
        articles.append(article_text)

        print("\n", title, url)
        print(article_summary, "\n")
        print(article_text)
        print(article_length)
        #input("ENTER\n")
