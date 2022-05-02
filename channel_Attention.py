

from tensorflow.keras.layers import Input, Reshape, Concatenate, Activation, RepeatVector, 
from tensorflow.keras.layers import Multiply, Lambda, Permute, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D



def attention_layer2(out, inputx):
  out = K.permute_dimensions(out, (0,1,3,2))
  inputx = K.permute_dimensions(inputx, (0,1,3,2))
  out_e = TimeDistributed(Dense(1, activation = 'tanh'))(out)
  # the softmax taking the above op
  e = TimeDistributed(Flatten())(out_e)
  # for scaled dot product uncomment the following line
  #e = Lamda(lambda values: values/K.sqrt(values.shape[-1]))
  a = TimeDistributed(Activation('softmax'))(e)
  temp = TimeDistributed(RepeatVector(1))(a*1.8)
  temp = TimeDistributed(Permute([2,1]))(temp)
  # multiply the weights with reshaped output of block 3
  output = (Multiply())([inputx, temp])
  # the attention adjusted output state
  output = TimeDistributed(Lambda(lambda values: K.sum(values, axis = 1)))(output)
  # expands the dimesnions across 2nd dimension of the feature map
  output = tensorflow.expand_dims(output, axis=2)
  return output




# define channel attention
def channel_attention(inputx, neurons_2, n, ratio=8):
    channel_axis = -1
    time_points = inputx.shape[1]
    # get channel
    channel = int(inputx.shape[channel_axis])
    # create dense layers
    Dense_one = TimeDistributed(Dense(channel//ratio, activation='elu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'))
    Dense_two = TimeDistributed(Dense(channel,activation='elu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'))
    # average pooling
    avg_pool = TimeDistributed(GlobalAveragePooling1D())(inputx[:,:,:n,:])
    avg_pool = TimeDistributed(Reshape((1,channel)))(avg_pool)
    assert avg_pool.shape[1:] == (time_points,1,channel)
    mlp_avg_pool = Dense_one(avg_pool)
    assert mlp_avg_pool.shape[1:] == (time_points,1,channel//ratio)
    mlp_avg_pool =  Dense_two(mlp_avg_pool)
    assert mlp_avg_pool.shape[1:] == (time_points,1,channel)
    # maxpooling
    max_pool = TimeDistributed(GlobalMaxPooling1D())(inputx[:,:,:n,:])
    max_pool = TimeDistributed(Reshape((1,channel)))(max_pool)
    assert max_pool.shape[1:] == (time_points,1,channel)
    mlp_max_pool = Dense_one(max_pool)
    assert mlp_max_pool.shape[1:] == (time_points,1,channel//ratio)
    mlp_max_pool = Dense_two(mlp_max_pool)
    assert max_pool.shape[1:] == (time_points,1,channel)
    # add the two features set created using max and avg pooling
    channel_attention_feature = Add()([mlp_avg_pool, mlp_max_pool])
    # apply multi-head attention to the channel information
    channel_attention_feature = attention_layer2(channel_attention_feature, inputx)
    channel_attention = TimeDistributed(Dense(neurons_2, activation='elu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'))(channel_attention_feature)
    return channel_attention

  
  
  
  
  
  
