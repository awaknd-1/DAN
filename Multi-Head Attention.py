
from tensorflow.keras.layers import Input, Reshape, Concatenate, Activation, RepeatVector, 
from tensorflow.keras.layers import Multiply, Lambda, Permute, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D



def feature_extraction(inputx, neurons_1, neurons_2):
  # extract features across electrode/ channels dimensions
  Cout = TimeDistributed(Dense(neurons_2, activation='elu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'))(inputx)
  # extract features across spectral dimensions
  F_inputs = K.permute_dimensions(Cout, (0,1,3,2))
  # apply dense
  Fout = TimeDistributed(Dense(neurons_1,activation='elu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'))(F_inputs)
  # swap the dimensions back to the orignal settings
  Fout = K.permute_dimensions(Fout, (0,1,3,2))
  return Fout

def MHDAN(inputx, neurons_1, neurons_2, n, h):
  # divide the vector into n heads, each for seprate attention 
  d2, d3 = inputx.shape[2], inputx.shape[3]
  out1 = list()
  # divide the vector into n heads, each for seprate attention 
  # spectral attention over multiple heads
  F1, F2, F3 = F_attention(inputx[:,:,:d2//h, :d3//h]), F_attention(inputx[:,:,d2//h:d2//h+d2//h, d3//h:d3//h+d3//h]), F_attention(inputx[:,:,d2//h+d2//h:, d3//h+d3//h:])
  # combine the heads across spectral dimension
  F = Concatenate(axis = 2)([F1, F2, F3])
  # attention across channels/electrodes
  C1, C2, C3 =  C_attention(inputx[:,:,:d2//h, :d3//h], n, ratio=8),  C_attention(inputx[:,:,d2//h:d2//h+d2//h, d3//h:d3//h+d3//h], n, ratio=8),  C_attention(inputx[:,:,d2//h+d2//h:, d3//h+d3//h:], n, ratio=8)
  # combine the heads across channel dimension
  C = Concatenate(axis = -1)([C1, C2, C3])
  # # combine the two attention maps
  Att_out = Multiply()([F, C])
  Att_out = K.permute_dimensions(Att_out, (0,1,3,2))
  # add the orignal feature map to attention feature map
  refined_out = Add()([inputx, Att_out])
  # apply fully connected layers to spectral and channel/electrodes dimension
  output = feature_extraction(refined_out, neurons_1, neurons_2)
  return output


