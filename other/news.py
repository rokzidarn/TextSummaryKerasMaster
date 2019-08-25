import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
import warnings
import tensorflow as tf
import os
import codecs
warnings.simplefilter(action='ignore', category=FutureWarning)


def create_tokenizer_from_hub_module(sess):
    bert_module = hub.Module("https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1")
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=False)


# MAIN

datapath = 'C:/Users/Rok/Downloads/news/DNEVNIK'
data = os.listdir(datapath)
i = 0
sess = tf.Session()
tokenizer = create_tokenizer_from_hub_module(sess)

for file in data:
    i += 1
    print(i)
    filepath = datapath + '/' + file

    try:
        with open(filepath) as fp:
            name = os.path.splitext(os.path.basename(fp.name))[0]

            arr = []
            next = True
            while next:
                line = fp.readline()
                if line.startswith('# Summary:'):
                    next = False

            summary = fp.readline()
            if len(summary) < 2:
                continue
            else:
                tmp = summary.split(' ')
                tmp.pop(0)  # TODO
                summary = ' '.join(tmp)

            fp.readline()

            line = fp.readline()
            arr.append(line)
            while line:
                line = fp.readline()
                arr.append(line)

            if len(arr) > 1 and arr[0].startswith('(Foto:'):
                arr.pop(0)

            if len(arr) > 2 and '@' in arr[-2]:
                arr.pop(-2)

            article = ''.join(arr)

            #print(summary, '\n', summary)

            article_tokens = tokenizer.tokenize(article)
            article_words_len = len(article.split(' '))
            article_token_len = len(article_tokens)

            summary_tokens = tokenizer.tokenize(summary)
            summary_words_len = len(summary.split(' '))
            summary_token_len = len(summary_tokens)

            #print(word_length, '\n', token_length)

            if article_token_len >= 200 and article_token_len <= 550 and summary_token_len >= 50 and summary_token_len <= 150:
                try:
                    with codecs.open('../data/news/articles/'+name+'.txt', 'w', encoding='utf8') as f:
                        f.write("{}\n".format(article))
                    with codecs.open('../data/news/summaries/'+name+'.txt', 'w', encoding='utf8') as f:
                        f.write("{}\n".format(summary))
                except:
                    continue
    except:
        continue
