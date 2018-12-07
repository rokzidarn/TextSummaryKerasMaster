import wikipediaapi
import pywikibot
from pywikibot import pagegenerators

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
    if limit == 0:
        break

    title = page.title()
    url = page.full_url()

    wiki_page = wiki.page(title)
    article_text = wiki_page.text
    article_summary = wiki_page.summary
    article_length = len(article_text.split())

    print(".", end="", flush=True)

    if 1000 < article_length < 2000:
        limit = limit - 1

        titles.append(title)
        urls.append(url)
        summaries.append(article_summary)
        articles.append(article_text)

        print("\n", title, url)
        print(article_summary[0:100])
        print(article_length)
        #input("ENTER\n")
