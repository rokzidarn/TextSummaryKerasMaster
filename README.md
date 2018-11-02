TEXT SUMMARIZATION
------------------
Python + BS4 + NTLK + Tensorflow + Keras

# word embeddings

    word embeddings are a type of word representation that allows words with similar meaning to have a similar representation
    words that have similar context will have similar meanings
    individual words are represented as real-valued vectors in a predefined vector space
    dense and low-dimensional vectors -> faster computation with neural networks
    
    algorithm:
        word ('hello') -> one-hot encoded (0,0,0,1,0,0,...) -> reavl-valued vector (0.312, -0.110, -0.499, 0.765, ...)
        word2vec: CBOW or skipgram, statistical method, CBOW -> predicts context based on current word
        glove: matrix factorization technique (LSA), context defined by word co-occurance matrix

    Keras:
        Embedding(
            input_dim='size_of_vocabulary', 
            output_dim='size_of_desired_vector_space', 
            input_length='number_of_words_in_document'
        ) 
        => returns learned weights, 2D vector with one embedding for each word in the input sequence of words,
            to connect to a Dense layer, you must add Flatten() after Embedding()
            
        model = Sequential()
        model.add(Embedding(1000, 64, input_length=10))  # 1000 possible values, 64D embedding       
        input_array = np.random.randint(1000, size=(32, 10))  # values [0...999], 32x10   
        model.compile('rmsprop', 'mse')
        output_array = model.predict(input_array)  # 32x10x64
          
# keras basics

    len(data)  # size of dataset, number of samples
    data.shape  # return tensor dimensions
    data.ndim  # returns dimensionality
    data.reshape()  # transforms tensor dimensions, (100, 8, 8) -> (100, 8*8)
    data.astype()  # changes data type
    to_categorical(labels)  # encodes labels to simplify predictions
    one_hot()  # vectorize, transform to zeros/one vector
    pad_sequences(x_train, maxlen=maxlen)  # pads, set fixed length on all of samples  
    reverse_sequences() => [x[::-1] for x in x_train] 
    
    network = models.Sequential()
    embedding_layer = Embedding(1000, 64, input_length=maxlen)  # 3D tensor, 1000 of possible words, 64D embeddings
    network.add(Flatten())  # flattens to shape (1000, 64*maxlen)
    
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))  # output = relu(dot(W, input) + b)
        # fully connected, 512 output units (dimension of tensor), next input has to expect 512 to work properly
    network.add(layers.Dropout(0.5))  # 50% of weights are discarded, (set to 0) to fight overfitting
    network.add(layers.Dense(10, activation='softmax'), kernel_regularizer=regularizers.l2(0.001))
        # regularizer L2 for overfitting problem
        
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        # loss = performance of the network, has to be minimized
        # optimizer = mechanism for updating weights, learning parameters based on loss function
        
    history = network.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_val, y_val))
        # training, 5 iterations, batches pf 128 per iteration
    history_dict = history.history  # data from learning, accuracy + loss metrics
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    
    predictions = model.predict(x_test)
    result = np.argmax(predictions[0])  # predicts class, highest probability
    
    network.summary()  # shows data of network, number of learable parameters
    network.add(SimpleRNN(32, return_sequences=True))  # if using multiple RNNs use attribure return_sequences=True
    network.add(layers.Bidirectional(layers.LSTM(32)))
    
    # functional API
    
    text_input = Input(shape=(None,), dtype='int32', name='text')
    embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)
    encoded_text = layers.LSTM(32)(embedded_text)
    
    question_input = Input(shape=(None,), dtype='int32', name='question')
    embedded_question = layers.Embedding( 32, question_vocabulary_size)(question_input)
    encoded_question = layers.LSTM(16)(embedded_question)
    
    concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
    answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)
    dense_model.add(layers.BatchNormalization())
    model = Model([text_input, question_input], answer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])