



# define the model
def define_model(trainX):
    # network-1
    _,a,b,c = trainX.shape
    inputs1 = Input(shape=(a,b,c))
    # Block-1
    out1 = MHDAN(inputs1,b,c, 15, 3)
    out = TimeDistributed(BatchNormalization(center = True, scale = True))(out1)
    out = Add()([inputs1, out])
    out = Activation('elu')(out)
    #out = conv_block(out,64, 3, 2)
    #out = residual_module(out,128)
    # maxpool
    #out = conv_block(out,128, 3, 2)
    out = feature_extraction(out,128,128)
    out1 = TimeDistributed(BatchNormalization(center = True, scale = True))(out)
    out = Add()([out, out1])
    out = Activation('elu')(out)
    # Block-2
    out = feature_extraction(out,50,50)
    out1 = TimeDistributed(BatchNormalization(center = True, scale = True))(out)
    out = Add()([out, out1])
    out = Activation('elu')(out)
    # block-3
    out = TimeDistributed(Flatten())(out)
    out = TimeDistributed(Dense(256,activation = 'elu'))(out)
    #embed = positional_encoding(a, 256)
    #out = Add()([out,embed])
    # the attention adjusted output state
    # the softmax taking the above op
    output = attention_layer(out)
    output = (Dense(128,activation = 'elu'))(output)
    # network-2
    outputs = (Dense(64, activation = 'elu'))(output)
    outputs = (Dense(2, activation = 'sigmoid'))(outputs)
    model = Model(inputs = inputs1, outputs = outputs)
    # compile
    optim = Adam(0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])
    return model
    

def model_fit(trainX, testX, trainy, testy, e, bs):
    model =  define_model(trainX)
    # fit the model on the datasets
    model.fit(trainX, trainy, epochs = e, batch_size = bs, shuffle = True, verbose = 0)
    # calcule the accuracy of the model
    _, acc = model.evaluate(testX, testy, batch_size = 64, verbose = 0)
    # shows whcih class was classified correctly
    yhat_classes = model.predict(testX, verbose=0)
    yhat_classes = np.argmax(yhat_classes, axis=1)
    del model
    return acc, yhat_classes
  
  
  
