import tensorflow
from tensorflow.keras.layers import Input, Reshape, Concatenate, Activation, RepeatVector, Multiply, Lambda, Permute, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Dropout, TimeDistributed, Dense, Flatten, BatchNormalization, Add
import tensorflow.keras.backend as K

class DAN():
    def __init__(self,*, beta):
        super(DAN, self).__init__()
        
        self.beta = beta


    def attention_layer(self, sp, inputs):
        out_e = TimeDistributed(Dense(1, activation = 'tanh'))(sp)
        # the softmax taking the above op
        e = TimeDistributed(Flatten())(out_e)
        #e = Lamda(lambda values: values/K.sqrt(values.shape[-1]))
        a = TimeDistributed(Activation('softmax'))(e)
        temp = TimeDistributed(RepeatVector(1))(a*self.beta)
        temp = TimeDistributed(Permute([2,1]))(temp)
        # multiply the weights with reshaped output of block 3
        output = (Multiply())([inputs, temp])
        # the attention adjusted output state
        output = TimeDistributed(Lambda(lambda values: K.sum(values, axis = 1)))(output)
        output = tensorflow.expand_dims(output, axis=-1)
        return output

    # used for spatial attention 
    def F_attention(self, inputs):
        sp_avg_pool = TimeDistributed(Lambda(lambda x: K.mean(x,axis=-1, keepdims=True)))(inputs)
        sp_max_pool = TimeDistributed(Lambda(lambda x: K.max(x,axis=-1, keepdims=True)))(inputs)
        sp = Concatenate(axis=-1)([sp_avg_pool, sp_max_pool])
        sp = self.attention_layer(sp, inputs)
        return sp

        # define channel attention
    def C_attention(self, inputx, n, ratio=8):
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
        channel_attention_feature = K.permute_dimensions(channel_attention_feature, (0,1,3,2))
        inputx = K.permute_dimensions(inputx, (0,1,3,2))
        # performs attention mechanism across channel dimension
        channel_attention_feature = self.attention_layer(channel_attention_feature, inputx)
        return channel_attention_feature

    def feature_extraction(self, inputx, neurons_1, neurons_2):
        # extract features across electrode/ channels dimensions
        Cout = TimeDistributed(Dense(neurons_2, activation='elu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'))(inputx)
        # extract features across spectral dimensions
        F_inputs = K.permute_dimensions(Cout, (0,1,3,2))
        # apply dense
        Fout = TimeDistributed(Dense(neurons_1,activation='elu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'))(F_inputs)
        # swap the dimensions back to the orignal settings
        Fout = K.permute_dimensions(Fout, (0,1,3,2))
        return Fout

    # define attention
    def temporal_attention(self,out):

      d2 = out.shape[-1]
      # the attention adjusted output state
      out_e = TimeDistributed(Dense(1, activation = 'tanh'))(out)
      # the softmax taking the above op
      e = TimeDistributed(Flatten())(out_e)
      a = Activation('softmax')(e)
      temp = TimeDistributed(RepeatVector(d2))(a*self.beta)
      temp = TimeDistributed(Permute([2,1]))(temp)
      # multiply the weights with reshaped output of block 3
      output = Multiply()([out, temp])
      # the attention adjusted output state
      output = TimeDistributed(Lambda(lambda values: K.sum(values, axis = 1)))(output)
      return output

    
    def MHDAN(self, inputx, neurons_1, neurons_2, n, h):
        # divide the vector into n heads, each for seprate attention 
        d2, d3 = inputx.shape[2], inputx.shape[3]
        # divide the vector into n heads, each for seprate attention 
        # spectral attention over multiple heads
        F1, F2, F3 = self.F_attention(inputx[:,:,:d2//h, :d3//h]), self.F_attention(inputx[:,:,d2//h:d2//h+d2//h, d3//h:d3//h+d3//h]), self.F_attention(inputx[:,:,d2//h+d2//h:, d3//h+d3//h:])
        # combine the heads across spectral dimension
        F = Concatenate(axis = 2)([F1, F2, F3])
        # attention across channels/electrodes
        C1, C2, C3 =  self.C_attention(inputx[:,:,:d2//h, :d3//h], n, ratio=8),  self.C_attention(inputx[:,:,d2//h:d2//h+d2//h, d3//h:d3//h+d3//h], n, ratio=8),  self.C_attention(inputx[:,:,d2//h+d2//h:, d3//h+d3//h:], n, ratio=8)
        # combine the heads across channel dimension
        C = Concatenate(axis = 2)([C1, C2, C3])
        C = K.permute_dimensions(C, (0,1,3,2))
        # # combine the two attention maps
        Att_out = Multiply()([F, C])
        Att_out = K.permute_dimensions(Att_out, (0,1,3,2))
        # add the orignal feature map to attention feature map
        refined_out = Add()([inputx, Att_out])
        # apply fully connected layers to spectral and channel/electrodes dimension
        output = self.feature_extraction(refined_out, neurons_1, neurons_2)
        return output



    


    