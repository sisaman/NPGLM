from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model


def train_encoder(X, epochs=50, latent_factor=2):
    n_samples, timesteps, input_dim = X.shape
    latent_dim = int(latent_factor * input_dim)

    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim)(inputs)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    sequence_autoencoder.compile(optimizer='adadelta', loss='mse')
    history = sequence_autoencoder.fit(X, X[:, ::-1, :], epochs=epochs, batch_size=1024, shuffle=True)
    
    print('Autoencoder Training Loss: %.4f' % history.history['loss'][-1])
    return encoder
