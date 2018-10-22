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
