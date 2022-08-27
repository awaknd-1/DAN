from Multi-Head-Attention import DAN

# define the model
def define_model(trainX):
    # network-1
    _,a,b,c = trainX.shape
    inputs1 = Input(shape=(a,b,c))
    # Block-1
    out1 = DAN(beta=1.8).MHDAN(inputs1,b,c, 15, 3)
    out = TimeDistributed(BatchNormalization(center = True, scale = True))(out1)
    out = Add()([inputs1, out])
    out = Activation('relu')(out)
    # maxpool
    out = DAN(beta=1.8).feature_extraction(out,128,128)
    out1 = TimeDistributed(BatchNormalization(center = True, scale = True))(out)
    out = Add()([out, out1])
    out = Activation('relu')(out)
    # Block-2
    out = DAN(beta=1.8).feature_extraction(out,50,50)
    out1 = TimeDistributed(BatchNormalization(center = True, scale = True))(out)
    out = Add()([out, out1])
    out = Activation('relu')(out)
    # block-3
    out = TimeDistributed(Flatten())(out)
    out = TimeDistributed(Dense(256,activation = 'relu'))(out)
    output = DAN(beta=1).temporal_attention(out)
    output = Flatten()(output)
    output = (Dense(128,activation = 'relu'))(output)
    # network-2
    outputs = (Dense(64, activation = 'relu'))(output)
    outputs = (Dense(2, activation = 'sigmoid'))(outputs)
    model = Model(inputs = inputs1, outputs = outputs)
    # compile
    optim = Adam(0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])
    return model


def model_fit(trainX, testX, trainy, testy):
    model =  define_model(trainX)
    # fit the model on the datasets
    model.fit(trainX, trainy, epochs = 30, batch_size = 32, shuffle = True, verbose = 1)
    #model.save('multi_freq_fusion.h5')
    # calcule the accuracy of the model
    _, acc = model.evaluate(testX, testy, batch_size = 32, verbose = 0)
    # shows whcih class was classified correctly
    yhat_classes = model.predict(testX, verbose=0)
    yhat_classes = np.argmax(yhat_classes, axis=1)
    del model
    return acc, yhat_classes