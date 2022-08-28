from Multi-Head-Attention import DAN
from tensorflow.keras.layers import TimeDistributed, BatchNormalization, Activation, Dense, Flatten, Add, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


class DANet():
    def __init__(self,*,beta): 
      super(DANet, self).__init__()
      self.beta = beta

    
    # define the model
    def define_model(self, trainX, N, heads, ff1, ff2, TDense, dense1, dense2, lr):
      # network-1
      _,a,b,c = trainX.shape
      inputs1 = Input(shape=(a,b,c))
      # Block-1
      out1 = DAN(beta = self.beta).MHDAN(inputs1, b, c, N, heads)
      out = TimeDistributed(BatchNormalization(center = True, scale = True))(out1)
      out = Activation('elu')(out)
      # maxpool
      out = DAN(beta = self.beta).feature_extraction(out, ff1, ff1)
      out1 = TimeDistributed(BatchNormalization(center = True, scale = True))(out)
      out = Add()([out, out1])
      out = Activation('elu')(out)
      # Block-2
      out = DAN(beta = self.beta).feature_extraction(out, ff2, ff2)
      out1 = TimeDistributed(BatchNormalization(center = True, scale = True))(out)
      out = Add()([out, out1])
      out = Activation('elu')(out)
      # block-3
      out = TimeDistributed(Flatten())(out)
      out = TimeDistributed(Dense(TDense, activation = 'elu'))(out)
      output = DAN(beta=1).temporal_attention(out)
      output = Flatten()(output)
      output = (Dense(dense1, activation = 'elu'))(output)
      # network-2
      outputs = (Dense(dense2, activation = 'elu'))(output)
      outputs = (Dense(2, activation = 'sigmoid'))(outputs)
      model = Model(inputs = inputs1, outputs = outputs)
      # compile
      optim = Adam(lr)
      model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])
      return model

