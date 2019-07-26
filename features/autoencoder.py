import time
import numpy as np
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def encode(X_list, epochs=50, latent_factor=2):
#     start_time = time.time()
    for i in range(1, len(X_list)):
        X_list[i] -= X_list[i - 1]
    scaler = MinMaxScaler(copy=False)
    for X in X_list:
        scaler.fit_transform(X)

    X = np.stack(X_list, axis=1)  # X.shape = (n_samples, timesteps, n_features)
    n_samples, timesteps, input_dim = X.shape
    latent_dim = int(latent_factor * input_dim)
#     x_train, x_test = train_test_split(X, stratify=Y)

    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim)(inputs)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    sequence_autoencoder.compile(optimizer='adadelta', loss='mse')
    history = sequence_autoencoder.fit(X, X[:,::-1,:], epochs=epochs, batch_size=256, shuffle=True)
    # sequence_autoencoder.save('autorencoder.model')
    print('Autoencoder Training Loss: %.4f' % history.history['loss'][-1])
#     print(time.time()-start_time)
    return encoder.predict(X)
