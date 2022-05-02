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
