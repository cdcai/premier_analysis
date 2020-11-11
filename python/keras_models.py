import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow.keras as keras

# %%
n_grams_setting = None
time_seq = 225
lstm_dropout = 0.2
# %% Demo model

# NOTE: We already split our text, so we don't need to tokenize
# Text vectorization pre-processing layer
text_to_vec = TextVectorization(split=None, ngrams=n_grams_setting)

text_to_vec.adapt(X)

# Get number of tokens
n_tok = len(text_to_vec.get_vocabulary())

# %% Model

input_layer = keras.Input(shape=(time_seq), dtype=tf.string, ragged=True)
# NOTE: Not sure if we need to mask ragged or not, but if so, there should
# be a masking layer here and the embedding layer should be set to ignore
# 0 as a mask and input_dim=n_tok+1 accordingly
emb_layer = keras.layers.Embedding(n_tok, output_dim=128, input_length=time_seq)(
    input_layer
)
lstm_layer = keras.layers.LSTM(250, use_bias=False, dropout=lstm_dropout)(emb_layer)
dense = keras.layers.Dense(250)(lstm_layer)
output_dim = keras.layers.Dense(1, activation="softmax")(dense)

model = keras.Model(input_layer, output_dim)

model.compile(optimizer="adam", loss="binary_cross_entropy", metrics="acc")