#! /usr/bin/env python 
import glob, os
import numpy as np
from ROOT import *

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler

# Keras imports
from keras.utils import np_utils


# Define this somewhere else?
btagthresh = 0.77


def Generator():
    """
    A generator to pass data to classifier 
    in managable chunks for training
    """
    #Just as a test for now.
    #Later thsi should be able to do this for a large amount of chains
    #That will be automatically created 


    sigfolder = "user.fschenck.303345.MadGraphPythia8EvtGen.DAOD_EXOT4.e4352_s2608_r7772_r7676_p2666.16-09-19_output.root/"
    ttbarfolder = "user.fschenck.410000.PowhegPythiaEvtGen.DAOD_EXOT4.e3698_s2608_s2183_r7725_r7676_p2719.16-09-09-v2_output.root/"
    sgtopfolder = "user.fschenck.410011.PowhegPythiaEvtGen.DAOD_EXOT4.e3824_s2608_s2183_r7725_r7676_p2719.16-09-09-v2_output.root/"
    wjetsfolder = "user.fschenck.363436.Sherpa.DAOD_EXOT4.e4715_s2726_r7725_r7676_p2708.16-09-09-v2_output.root/"


    folderlist = [sigfolder, ttbarfolder, sgtopfolder, wjetsfolder] 

    chainlist = []
    for fold in folderlist:
        chain = TChain("nominal")
        chain.SetBranchStatus("*", 0)
        chain.SetBranchStatus("el_*", 1)
        chain.SetBranchStatus("mu_*", 1)
        chain.SetBranchStatus("jet_*", 1)
        chain.SetBranchStatus("met_*", 1)
        
    
        #print fold
        for file in os.listdir("../data/" + fold):
            chain.Add("../data/" + fold + "/" + file)    
        #    print "adding " + file
        chainlist.append(chain)


	runevents = chainlist[0].GetEntries()
	print "Events: " + str(runevents)	

    nsec = 100  
    while True:
        for i in range(runevents/nsec): # 3 is just a test value for now
                    
            dset = []
            for t in chainlist:
                #print chainlist.index(t)
                for j in range(nsec):
                        t.GetEntry(i*nsec + j)
                           #print "Entry no: " + str(i*nsec + j)

                        # Assumes one lepton in final state
                        if (t.el_e.size() != 0):
                            l = [t.el_e[0], t.el_pt[0], t.el_phi[0], t.el_eta[0], t.el_charge[0], 0]
                        elif (t.mu_e.size() != 0):
                            l = [t.mu_e[0], t.mu_pt[0], t.mu_phi[0], t.mu_eta[0], t.mu_charge[0], 1]
                        else: 
                            continue

                        #Just for testing, remove. Find out what negative btag values mean
                        if True:    
                        #if ((t.jet_mv2c10[0] > btagthresh) != (t.jet_mv2c10[1] > btagthresh)):

                            if (t.jet_mv2c10[0] > btagthresh):
                                bj = [t.jet_e[0], t.jet_pt[0], t.jet_phi[0], t.jet_eta[0], 1]
                                lj = [t.jet_e[1], t.jet_pt[1], t.jet_phi[1], t.jet_eta[1], 0]
                            else:
                                lj = [t.jet_e[0], t.jet_pt[0], t.jet_phi[0], t.jet_eta[0], 0]
                                bj = [t.jet_e[1], t.jet_pt[1], t.jet_phi[1], t.jet_eta[1], 1]

                        else:
                            #print t.jet_mv2c10[0]
                            #print t.jet_mv2c10[1]
                            continue


                        MET = [t.met_met, t.met_phi]
                        #print MET

                        data = l + bj + lj + MET
                        dset = dset + data + [chainlist.index(t)]

            #now add training labels
            dset = np.array(dset).reshape(len(dset)/19, 19)

            X = dset[:,:18]
            Y = dset[:,18]

            scaler = MinMaxScaler(feature_range=(0, 1))
            reX = scaler.fit_transform(X)
    
            reY = np_utils.to_categorical(Y)
            yield [reX, reX], reY


    

if __name__ == '__main__':
    g = Generator()
    for X, Y in g:
        pass


