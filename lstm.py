import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Dropout, CuDNNLSTM
from keras.callbacks import EarlyStopping

def kezhNet(n_channels,n_features,n_units_input=100,n_units_output=1):
    model = Sequential()
    model.add(LSTM(n_units_input, activation='relu',input_shape=(n_channels, n_features)))
    #model.add(CuDNNLSTM(n_units_input, input_shape=(n_channels, n_features)))
    model.add(Dense(n_units_output))
    return model

class LSTM_Model():
    def __init__(self,n_channels,n_features,learning_rate,model_path,n_units_input=100,n_units_output=1):
        self.multi_model = kezhNet(n_channels,n_features,n_units_input,n_units_output)
        self.loss = 'mse'
        self.optimizer = keras.optimizers.Adam(lr=learning_rate)
        self.model_path = model_path
        self.train_loss = []
        self.test_loss = []

    def trainModel(self,train_X,train_y,test_X,test_y,epochs,batch_size,patience):
        self.multi_model.compile(loss=self.loss, optimizer=self.optimizer)
        earlyStop = EarlyStopping(monitor="val_loss",verbose=2,mode='min',patience=patience,restore_best_weights=True)
        multi_history = self.multi_model.fit(train_X, train_y,batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_y), verbose=2,callbacks=[earlyStop], shuffle=False)
        self.train_loss = multi_history.history['loss']
        self.test_loss = multi_history.history['val_loss']
        self.multi_model.save(self.model_path)
        print("Model saved in file: %s" % self.model_path)

    def makePredictions(self,test_X):
        multi_model = load_model(self.model_path)
        print("Model restored from file: %s" % self.model_path)
        return multi_model.predict(test_X)
