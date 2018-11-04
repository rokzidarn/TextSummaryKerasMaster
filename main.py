# ENCODER-DECODER

# the encoder-decoder architecture is a way of organizing recurrent neural networks
# for sequence prediction problems that have a variable number of inputs

# encoder: The encoder reads the entire input sequence and encodes it into an internal representation,
# often a fixed-length vector called the context vector (capturing the meaning of the source document)
# decoder: The decoder reads the encoded input sequence from the encoder and generates the output sequence,
# the decoder must generate each word in the output sequence given two sources of information
    # context Vector: The encoded representation of the source document provided by the encoder
    # generated Sequence: The word or sequence of words already generated as a summary

# an extension of the encoder-decoder architecture (attention) is to provide a more expressive form of the encoded
# input sequence and allow the decoder to learn where to pay attention to the encoded input when generating
# each step of the output sequence

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
