def mapper_model(input_shape=(192,), latent_dim=8):
   # Dropout einbauen
    #layer dicker machen
    #activations ausprobieren
    
    mapper_inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(mapper_inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(latent_dim, activation='relu')(x)
    
    return Model(mapper_inputs, x)
mapper = mapper_model()
mapper.summary()