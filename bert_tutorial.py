import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tqdm import tqdm_notebook
from tensorflow.keras import backend as K

sess = tf.Session()
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = 256


def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)


def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


def download_and_load_datasets(force_download=False):
  dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz",
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
      extract=True)

  train_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
  test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))

  return train_df, test_df

tf.logging.set_verbosity(tf.logging.ERROR)
train_df, test_df = download_and_load_datasets()


train_text = train_df['sentence'].tolist()
train_text = [' '.join(t.split()[0:max_seq_length]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = train_df['polarity'].tolist()


test_text = test_df['sentence'].tolist()
test_text = [' '.join(t.split()[0:max_seq_length]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df['polarity'].tolist()

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    print("bert hub")
    bert_module = hub.Module(bert_path)
    print("bert tokenization")
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    print("bert session")
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


tokenizer = create_tokenizer_from_hub_module()


def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in examples:
        input_id, input_mask, segment_id, label = convert_single_example(tokenizer, example, max_seq_length)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)

    return (np.array(input_ids), np.array(input_masks), np.array(segment_ids), np.array(labels).reshape(-1, 1),)


def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []

    for text, label in zip(texts, labels):
        InputExamples.append(InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label))

    return InputExamples


texts = [
 ["One of the all-time great science fiction works, as visionary and thought-provoking as Blade Runner or even Gilliam's own Brazil. Willis gives his best performance here, but he's outdone by Pitt's incredibly frenetic turn that's unlike anything he's done before or since. Even Stowe isn't out of her league here, though. The story is very layered and offers quite a lot to think about. The climactic scene is beautifully magnificent, and the last lines fit perfectly. The scenes in the mental hospital are creepy and yet so funny in their own way. Lots of dark humour on display here. Fantastic production design and suitably bizarre cinematography. In my top ten."],
 ["Proof why Hollywood conventions are in place. Stale dialogue, underdeveloped and flat characters and a disjointed storyline are only part of the problems with this gangster classic wannabe. An attempt to be daring and different but this appears to be a slap-together attempt at recreating the magic of Arthur Penn 's Bonnie and Clyde (1967) and George Roy Hill 's Butch Cassidy and the Sundance Kid (1969)- truly innovative filmmakers and films - but falling well below the bar. Problems with storylines being self-explanatory result in the need for a voiceover to explain problem sections. The editing appears again to be an attempt to duplicate the previous classics but is occasionally disjointed and cause more problems for me technically. Unnecessary shots are thrown in to justify the filming of them but would have better served the viewer by sitting on the cutting room floor. Stills, black & white montages and period music are thrown in from time to time in attempts to either be different or to cover up for scenes that can't transition well or to replace scenes that just didn't work at all and again are reminiscent of Butch Cassidy and the Sundance Kid (1969).<br /><br />Overly dramatic pauses between sentences, random shots of surrounding scenery that wasn't needed for storytelling plus over-the-top acting of bit players and supporting actors was reminiscent of the backyard camcorder directors of the late 1980's - I was left wondering who was in charge of this film during production and during post-production. The playing of music in most"],
 ['This is an entertaining look at the Gospel as presented by Johnny Cash (adorned in black, of course) who sings a lot and narrates a bit also. If you like Johnny Cash, this film is quite enjoyable. Also note the blonde depiction of Jesus in this work...just for fun, try to think of five Jewish men who have blonde hair...? Anyway, its a fun presentation of the greatest and most important story of all.'],
]
sentiments = [4, 10, 10]
labels = [0, 1, 1]

# Convert data to InputExample format
train_examples = convert_text_to_examples(texts, labels)
print("tokenizer")
#test_examples = convert_text_to_examples(test_text[:10], test_label[:10])

# Convert to features
(train_input_ids, train_input_masks, train_segment_ids, train_labels) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)
# (test_input_ids, test_input_masks, test_segment_ids, test_labels) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)


class BertLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )

        trainable_vars = self.bert.variables
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size


def build_model(max_seq_length):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
    dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


model = build_model(max_seq_length)

# Instantiate variables
initialize_vars(sess)

model.fit(
    [train_input_ids, train_input_masks, train_segment_ids],
    train_labels,
    epochs=5,
    batch_size=32
)
