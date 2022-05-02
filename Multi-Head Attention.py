# performs multi-head attention on the spectro-spatial feature map.
def multi_head_attention(inputx, out, num_head, Atype):
  # divide the vector into n heads, each for seprate attention 
  d1 = tensorflow.shape(out)[0]
  out1 = list()
  # spatial or channel
  if Atype == "Spectral": 
    # divide the vector into n heads, each for seprate attention 
    new_out = tensorflow.reshape(out, shape= (d1,out.shape[1],out.shape[2]//num_head, out.shape[3], num_head))
    output11 = attention_layer2(new_out[:,:,:,:,0], inputx[:,:,:out.shape[2]//num_head,:])
    output12 = attention_layer2(new_out[:,:,:,:,1], inputx[:,:,out.shape[2]//num_head:,:])
    output1 = Add()([output11, output12])
    out1.append(output1)
    output = Concatenate(axis=3)(out1)
    output = tensorflow.reshape(output, shape = (d1, out.shape[1], 1, out.shape[2]))
  elif Atype == "Channel": 
    new_out = tensorflow.reshape(out, shape= (d1,out.shape[1],out.shape[2],out.shape[3]//num_head, num_head))
    output11 = attention_layer2(new_out[:,:,:,:,0], inputx[:,:,:,:out.shape[3]//num_head])
    output12 = attention_layer2(new_out[:,:,:,:,1], inputx[:,:,:,out.shape[3]//num_head:])
    output1 = Add()([output11, output12])
    out1.append(output1)
    output = Concatenate(axis=3)(out1)
    output = tensorflow.reshape(output, shape= (d1, out.shape[1], out.shape[2], out.shape[3]))
  else:
    raise ValueError('A very specific bad thing happened.')
  return output
