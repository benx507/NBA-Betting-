import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.neural_network import MLPClassifier, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, Activation, concatenate
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from keras import regularizers

data_path = "20182019last5.csv"

def load_data(data_csv):
        """
        Load the data from a csv file. Just return the X and y, splitting into
        train and test data will be handled elsewhere.
        """
        data = pd.read_csv(data_csv, index_col=0)

        data['outcome'] = data.apply(lambda x: 0 if x['winner'] == 'Away' else 1, axis=1)

        y = data['outcome']
        X = data.drop(['winner', 'outcome','winning_abbr'], axis=1)
        X = X.fillna(0)
        return X.to_numpy(), y.to_numpy()

def find_best_parameters():
    x,y= load_data(data_path)

    pca = PCA(n_components=46)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents)
    X_train, X_test, y_train, y_test = train_test_split(principalDf, y, test_size=0.25, random_state=42)
    sc_X = StandardScaler()
    X_trainscaled=sc_X.fit_transform(X_train)
    X_testscaled=sc_X.fit_transform(X_test)

    clf = MLPClassifier(max_iter=200)

    parameter_space = {
    'hidden_layer_sizes': [(100,), (50,100,), (50,75,100,), (200),(100,200),(100,150,200)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    }
    clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=5)

    clf.fit(X_trainscaled, y_train)

    print('Best parameters found:\n', clf.best_params_)
    y_pred = clf.predict(X_testscaled)
    print('Results on the test set:')
    print('Train Accuracy : %.3f'%clf.best_estimator_.score(X_trainscaled, y_train))
    print('Test Accuracy : %.3f'%clf.best_estimator_.score(X_testscaled, y_test))
    print('Best Accuracy Through Grid Search : %.3f'%clf.best_score_)
    print('Best Parameters : ',clf.best_params_)


hidden_layer = (100,150,200)
act = 'tanh'
sol = 'sgd'
learn = 'adaptive'
alp= 0.0001
iter = 500

def train_best_model(hidden_layer, act, sol, alp, learn, iter):
    x,y= load_data(data_path)

    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents)
    X_train, X_test, y_train, y_test = train_test_split(principalDf, y, test_size=0.25, random_state=42)
    sc_X = StandardScaler()
    X_train=sc_X.fit_transform(X_train)
    X_test=sc_X.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=iter, activation=act, solver=sol, alpha = alp, learning_rate=learn)
    
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 100
    N_BATCH = 128
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            clf.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(clf.score(X_train, y_train))

        # SCORE TEST
        scores_test.append(clf.score(X_test, y_test))

        epoch += 1

    plt.plot(scores_train, color='green', alpha=0.8, label='Test')
    plt.plot(scores_test, color='magenta', alpha=0.8, label='Train')
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.show()

train_best_model(hidden_layer, act, sol, alp, learn, iter)

#using a keras tensorflow model for deeper learning capability and more customization
def advanced_nn():
    x,y= load_data(data_path)
    pca = PCA(n_components=10)
    x = pca.fit_transform(x)
    x = tf.keras.utils.normalize(x, axis=1)
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu6,kernel_regularizer=regularizers.l2(0.01) ), )
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu6))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu6))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    #earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=500, validation_split=0.1, batch_size=32,)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

# advanced_nn()
