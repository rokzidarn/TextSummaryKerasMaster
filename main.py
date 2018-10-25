from keras.layers import Dropout
from keras import layers
from keras import Input
from keras.models import Model
import matplotlib.pyplot as plt

def plot_acc(history_dict, epochs):
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    fig = plt.figure()
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Testing acc')
    plt.title('Training and testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #fig.savefig('cpu_test.png')

# params
epochs = 40
dropout_rate = 0.05
embedding_size_text = 112
embedding_size_question = 96
latent_size_text = 142
latent_size_question = 128

"""

# model
text_input = Input(shape=(max_len_instance,))
embedded_text = layers.Embedding(vocabulary_size, embedding_size_text)(text_input)
encoded_text = layers.LSTM(latent_size_text)(embedded_text)
encoded_text = Dropout(dropout_rate)(encoded_text)

question_input = Input(shape=(max_len_question,))
embedded_question = layers.Embedding(vocabulary_size, embedding_size_question)(question_input)
encoded_question = layers.LSTM(latent_size_question)(embedded_question)
encoded_question = Dropout(dropout_rate)(encoded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
answer = layers.Dense(vocabulary_size_answers, activation='softmax')(concatenated)
answer = Dropout(dropout_rate)(answer)

model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# training
history = model.fit([Xitrain, Xqtrain], [Ytrain], batch_size=128, epochs=epochs, validation_data=([Xitest, Xqtest], [Ytest]))

history_dict = history.history  # data during training, history_dict.keys()
print("Max validaton acc: ", round(max(history_dict['val_acc']), 3))

gprah_epochs = range(1, epochs + 1)
plot_acc(history_dict, gprah_epochs)
"""