
from tensorflow.keras.layers import Input, Reshape, Concatenate, Activation, RepeatVector, Multiply, Lambda, Permute, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Dropout, TimeDistributed, Dense, Flatten, BatchNormalization, Add


# used for spatial attention 
def attention_layer1(out, inputx):
  out_e = TimeDistributed(Dense(1, activation = 'tanh'))(out)
  # the softmax taking the above op
  e = TimeDistributed(Flatten())(out_e)
  #e = Lamda(lambda values: values/K.sqrt(values.shape[-1]))
  a = TimeDistributed(Activation('softmax'))(e)
  temp = TimeDistributed(RepeatVector(1))(a*1.8)
  temp = TimeDistributed(Permute([2,1]))(temp)
  # multiply the weights with reshaped output of block 3
  output = (Multiply())([inputx, temp])
  # the attention adjusted output state
  output = TimeDistributed(Lambda(lambda values: K.sum(values, axis = 1)))(output)
  output = tensorflow.expand_dims(output, axis=-1)
  return output



# define channel attention
def F_attention(inputx):
    # now calculate average and max value across spatial dimensions
    spatial_avg_pool = TimeDistributed(Lambda(lambda x: K.mean(x,axis=-1, keepdims=True)))(inputx)
    spatial_max_pool = TimeDistributed(Lambda(lambda x: K.max(x,axis=-1, keepdims=True)))(inputx)
    sp = Concatenate(axis=-1)([spatial_avg_pool, spatial_max_pool])
    # apply multi head attention
    #spatial_attn = multi_head_attention(inputx, spatial_attn, 2, "Spectral")
    sp = attention_layer1(sp, inputx)
    return sp




