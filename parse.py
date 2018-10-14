from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTText

parser = PDFParser(open('../../Data/3.pdf', 'rb'))
doc = PDFDocument()
parser.set_document(doc)
doc.set_parser(parser)
doc.initialize('')

rsrcmgr = PDFResourceManager()
laparams = LAParams(line_margin=0.4, word_margin=0.3)
device = PDFPageAggregator(rsrcmgr, laparams=laparams)
interpreter = PDFPageInterpreter(rsrcmgr, device)

finish = False
summary_page = False

first = []
ex_summary = []
ex_title = []

for count, page in enumerate(doc.get_pages()):
    interpreter.process_page(page)
    layout = device.get_result()
    for element in layout:
        if count == 0 and isinstance(element, LTText):
            first.append(element.get_text())
        elif isinstance(element, LTText) and 'Povzetek' in element.get_text() and not summary_page:
            summary_page = True
        elif isinstance(element, LTText) and 'Povzetek' in element.get_text() and summary_page:
            finish = True
        if finish:
            ex_summary.append(element.get_text())

    if finish:
        break

ex_title = first[2:5]
if 'DIPLOMSKO DELO\n' in ex_title:
    ex_title.remove('DIPLOMSKO DELO\n')

if 'Povzetek\n' in ex_summary:
    ex_summary.remove('Povzetek\n')
if 'Naslov:' in ex_summary[0][:7]:
    del ex_summary[0]
if 'Avtor:' in ex_summary[0][:6]:
    del ex_summary[0]

words = []
for line in ex_title:
    words.append(line.split())

title = [item if 'ˇ' not in item else item.replace('ˇ', "") for sublist in words for item in sublist]
print(" ".join(title).upper())

words = []
for line in ex_summary:
    words.append(line.split())

summary = [item.replace('.', "").lower() if 'ˇ' not in item else item.replace('ˇ', "").replace('.', "").lower()
           for sublist in words for item in sublist]

try:
    index = summary.index("besede:")
    summary = summary[:index-1]
except ValueError:
    summary = summary

print(summary)
