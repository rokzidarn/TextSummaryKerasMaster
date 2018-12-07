import wikipediaapi
import pywikibot
from pywikibot import pagegenerators

wiki = wikipediaapi.Wikipedia('sl')
site = pywikibot.Site()
category = pywikibot.Category(site, 'Category:Naravoslovje')

generated = pagegenerators.CategorizedPageGenerator(category, recurse=2)
pages = []
num = 5

for page in generated:
    title = page.title()
    text = page.text
    text_length = len(text.split())

    if 1000 < text_length < 2000:
        print(title)
        pages.append(title)
        page_py = wiki.page(title)

        print(page_py.text[0:60])

        num = num - 1
        if num == 0:
            break
