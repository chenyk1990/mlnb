from keras.models import Sequential
from keras.layers import Activation, Dense, RepeatVector, Input
from keras.layers import RNN
from keras.src import ops
from sklearn.utils import shuffle
import numpy as np

"""
An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

By default, the JZS1 recurrent neural network is used
JZS1 was an "evolved" recurrent neural network performing well on arithmetic benchmark in:
"An Empirical Exploration of Recurrent Network Architectures"
http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.


Two digits inverted:
+ One layer JZS1 (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits inverted:
+ One layer JZS1 (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs


Four digits inverted:
+ One layer JZS1 (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs


Five digits inverted:
+ One layer JZS1 (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
"""

# First, let's define a RNN Cell, as a layer subclass.
import keras
class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = ops.matmul(inputs, self.kernel)
        output = h + ops.matmul(prev_output, self.recurrent_kernel)
        return output, [output]
        
class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    """
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
            
#         print('X.shape',X.shape,'X.type',type(X))
        return ''.join(self.indices_char[x] for x in X)
        
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset
TRAINING_SIZE = 50000
DIGITS = 3
INVERT = True
# Try replacing JZS1 with LSTM, GRU, or SimpleRNN
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
MAXLEN = DIGITS + 1 + DIGITS

chars = '0123456789+ '
ctable = CharacterTable(chars, MAXLEN)

questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    # Skip any addition questions we've already seen
    # Also skip any such that X+Y == Y+X (hence the sorting)
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    # Answers can be of maximum size DIGITS + 1
    ans += ' ' * (DIGITS + 1 - len(ans))
    if INVERT:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

print('Vectorization...')
X = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool_)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool_)
for i, sentence in enumerate(questions):
    X[i] = ctable.encode(sentence, maxlen=MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, maxlen=DIGITS + 1)

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
X, y = shuffle(X, y)
# Explicitly set apart 10% for validation data that we never train over
split_at = len(X) - len(X) / 10
(X_train, X_val) = (X[0:int(split_at),:], X[int(split_at):,:])
(y_train, y_val) = (y[:int(split_at)], y[int(split_at):])

print('Build model...')
model = Sequential()
model.add(Input(shape=(None,12))); #added
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# model.add(RNN(len(chars), HIDDEN_SIZE))
# x = keras.Input((None, 5))
# cell = MinimalRNNCell(HIDDEN_SIZE)
cell=[MinimalRNNCell(12), MinimalRNNCell(32), MinimalRNNCell(12)]
model.add(RNN(cell))

# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(DIGITS + 1))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
#     model.add(RNN(HIDDEN_SIZE, HIDDEN_SIZE, return_sequences=True))
    model.add(RNN(cell, return_sequences=True))
# For each of step of the output sequence, decide which character should be chosen
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=1, validation_data=(X_val, y_val), verbose=1)
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict(rowX, verbose=0)
        q = ctable.decode(rowX[0])
        correct = ctable.decode(rowy[0])
#         guess = ctable.decode(preds[0], calc_argmax=False)
        guess = ctable.decode(preds[0], calc_argmax=True)
        print('Q', q[::-1] if INVERT else q)
        print('T', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        print('---')
