#! /usr/bin/env python


from ROOT import *
from generator import Generator


# Keras includes
from keras.models import Sequential
from keras.layers import Dense, Merge




def main():
    
    model = Sequential()

    lh = Sequential()
    rh = Sequential()
    lh.add(Dense(36,input_dim=18, activation='tanh'))
    rh.add(Dense(36,input_dim=18, activation='relu'))

    merged = Merge([lh, rh], mode='concat')

    model.add(merged)
    #model.add(Dense(36, input_dim=18, init='uniform', activation='relu'))

    model.add(Dense(100, init='uniform', activation='relu'))
    model.add(Dense(100, init='uniform', activation='softmax'))
    model.add(Dense(100, init='uniform', activation='relu'))

    #model.add(Dense(72, init='uniform', activation='relu'))
        #model.add(Dense(72, init='uniform', activation='softmax'))
        #model.add(Dense(72, init='uniform', activation='relu'))


    model.add(Dense(4, init='normal', activation= 'softmax')) #'softmax')) #'sigmoid'

    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    
    gen = Generator()

    model.fit_generator(gen, samples_per_epoch = 80000, nb_epoch = 5 )    



if __name__ == '__main__':
    main()
