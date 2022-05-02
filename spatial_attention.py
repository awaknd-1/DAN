
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




# define spatial attention 
def spatial_attention(inputx, channel_attention, neurons_1, spatial_axis, time_points):
    # this makes both the matrices of same dimensions
    inputx = TimeDistributed(Dense(channel_attention.shape[-1],activation='elu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'))(inputx)
    # multiply the channel features with the orignal features.
    channel_attention = Multiply()([channel_attention, inputx])
    # now calculate average and max value across spatial dimensions
    spatial_avg_pool = TimeDistributed(Lambda(lambda x: K.mean(x,axis=-1, keepdims=True)))(channel_attention)
    spatial_max_pool = TimeDistributed(Lambda(lambda x: K.max(x,axis=-1, keepdims=True)))(channel_attention)
    spatial_attn = Concatenate(axis=-1)([spatial_avg_pool, spatial_max_pool])
    spatial_attn = attention_layer1(spatial_attn, inputx)
    # dense layers for feature extraction and reduction
    spatial_attn = TimeDistributed(Dense(neurons_1,activation='elu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'))(spatial_attn)
    spatial_attn = K.permute_dimensions(spatial_attn, (0,1,3,2))
    return spatial_attn




