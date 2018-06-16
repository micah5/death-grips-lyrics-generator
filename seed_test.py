from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
import numpy as np
import random
import sys
import io

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

with io.open('lyrics.txt', encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

diversity = 0.2
maxlen = 40
model = load_model('model.keras')
generated = ''
word = "bald head girl "
maxlen_word = (word * ((maxlen/len(word))+1))[:maxlen]
seed = maxlen_word#"cum cum cum cum cum cum cum cum cum cum "
generated += seed
print('----- Generating with seed: "' + seed + '"')
sys.stdout.write(generated)

for i in range(1000):
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(seed):
        x_pred[0, t, char_indices[char]] = 1.

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]

    generated += next_char
    seed = seed[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()
print()
