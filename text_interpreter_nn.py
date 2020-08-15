from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.layers import concatenate

class TextInterpreterNN:

    def __init__(self,input_size,vocab_size,embedding_dim):
        self.input_size = input_size
        self.text_interpreter_input = layers.Input(shape=(input_size,),name='main_input')
        self.text_interpreter_model = layers.Embedding(vocab_size,embedding_dim) (self.text_interpreter_input)
        self.text_interpreter_model = layers.Bidirectional(layers.LSTM(embedding_dim)) (self.text_interpreter_model)
        #self.text_interpreter_model = layers.Dense(64, activation= activations.linear)(self.text_interpreter_model)
        out = layers.Dense(3, activation=activations.softmax)(self.text_interpreter_model)
        self.model = models.Model(self.text_interpreter_input,out)


    def get_model(self,model_to_compile=None):
        if model_to_compile:
            self.model = model_to_compile
        optimizer = optimizers.SGD(lr=0.01)
        self.model.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
        return self.model

    def insert_emoji_feature(self,input_size):

        input = layers.Input(shape=(input_size,),name='emoji_feature_input')

        concatenated = concatenate([self.text_interpreter_model, input])        
        feature_model = layers.Dense(64,activation=activations.linear)(concatenated)

        out = layers.Dense(3, activation=activations.softmax)(feature_model)
        self.model = models.Model([self.text_interpreter_input,input],out)

