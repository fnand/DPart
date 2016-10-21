#! /usr/bin/env python


from ROOT import *
from generator import Generator
from make_model import seq_layer





def main():
    


    model = seq_layer()
    gen = Generator()

    model.fit_generator(gen, samples_per_epoch = 80000, nb_epoch = 5 )    


	# serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
            json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")




if __name__ == '__main__':
    main()
