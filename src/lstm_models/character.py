from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM

from lstm_models import ParaCompletionModel

class CharacterLSTMModel(ParaCompletionModel):
    'LSTM Model with characters  as inputs/outputs'