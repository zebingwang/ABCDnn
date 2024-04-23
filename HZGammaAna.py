from ABCD_dnn_mmd import ABCDdnn
import numpy as np
import matplotlib.pyplot as plt
import uproot
from onehotencoder import OneHotEncoder_int
import os
import pandas as pd
from root_pandas import *
import pickle

#featurevars = ['met', 'ht', 'pt5', 'pt6', 'njet', 'nbtag']

#rootfile='ttjjresult.root'

#featurevars = ['H_mass', 'H_pt', 'Z_pt', 'gamma_pt','gamma_mvaID', 'n_jets', 'n_leptons', 'regions']
#featurevars = ['H_mass', 'H_pt', 'H_eta', 'Z_pt', 'Z_eta', 'Z_mass', 'gamma_pt', 'gamma_eta', 'Z_lead_lepton_pt', 'Z_lead_lepton_eta', 'Z_sublead_lepton_pt', 'Z_sublead_lepton_eta', 'regions']
#featurevars = ['H_mass', 'H_relpt', 'l1g_deltaR', 'l2g_deltaR', 'Z_cos_theta', 'regions']
#featurevars = ['H_mass', 'H_relpt', 'Z_cos_theta', 'lep_cos_theta', 'BDT_score', 'regions']
featurevars = ['H_mass', 'BDT_score', 'H_relpt', 'Z_cos_theta', 'lep_cos_theta', 'Z_pt', 'gamma_pt', 'regions']
#featurevars = ['H_mass', 'Z_pt', 'gamma_pt', 'n_jets', 'n_leptons', 'regions']
rootfile_target='./datasets/data_inclusive.root'
rootfile_source='./datasets/ZGToLLG_inclusive.root'

def prepdata(filename='', treename='inclusive'):
    hzg = uproot.open(filename)
    hzgtree = hzg[treename]
    iscategorical = [False, False, False, False, False, False, False, True]
    #upperlimit = [170, 100, 1.0, 3]
    #lowerlimit = [105, 0, -1.0, 0]
    _onehotencoder = OneHotEncoder_int(iscategorical)

    arrays = hzgtree.arrays(featurevars, library="pd")
    inputtmp = pd.DataFrame(arrays)
    #inputtmp = inputtmp[(inputtmp["H_mass"]>105) & (inputtmp["H_mass"]<170)]

    inputnumpy = inputtmp.to_numpy(dtype=np.float32)
    inputs = _onehotencoder.encode(inputnumpy)
    ncats = _onehotencoder.ncats
    ncat_per_feature = _onehotencoder.categories_per_feature

    meanslist = []
    sigmalist = []
    currentcolumn = 0
    for ifeat, ncatfeat in zip(range(inputtmp.shape[1]), ncat_per_feature):
        if ncatfeat == 0: # fir float features, get mean and sigma
            mean = np.mean(inputnumpy[:, currentcolumn], axis=0, dtype=np.float32).reshape(1,1)
            sigma = np.std(inputnumpy[:, currentcolumn], axis=0, dtype=np.float32).reshape(1,1)
            #if sigma[0][0] < 2.0:
            #    mean = np.zeros(shape=(1, 1), dtype=np.float32)
            #    sigma = np.ones(shape=(1, 1), dtype=np.float32)
            meanslist.append(mean)
            sigmalist.append(sigma)
            currentcolumn += 1
        else: # categorical features do not get changed
            mean = np.zeros(shape=(1, ncatfeat), dtype=np.float32) 
            meanslist.append(mean)
            sigma = np.ones(shape=(1, ncatfeat), dtype=np.float32)
            sigmalist.append(sigma)
            currentcolumn += ncatfeat

    inputmeans = np.hstack(meanslist)
    inputsigma = np.hstack(sigmalist)

    normedinputs = (inputs-inputmeans) / inputsigma

    return inputtmp, normedinputs, inputmeans, inputsigma, ncat_per_feature

def writetorootfile(rootfilename, datadict):
    branchdict = {}
    for key, data in datadict.items():
        branchdict[key] = data.dtype
    #tree = uproot.newtree(branches=branchdict)
    tree = branchdict
    with uproot.recreate(rootfilename) as f:
        f['mytree'] = tree
        f['mytree'].extend(datadict)

    pass

def train_and_validate(steps=10000, minibatch=128, LRrange=[0.0001, 0.00001, 10000, 0], beta1=0.9, beta2=0.999, nafdim=16, depth=2, \
    savedir='abcdnn', seed=100, retrain=False, train=True, permute=True):
    #rawinputs, normedinputs, inputmeans, inputsigma, ncat_per_feature = prepdata()
    rawinputs_target, normedinputs_target, inputmeans_target, inputsigma_target, ncat_per_feature_target = prepdata(filename=rootfile_target)
    rawinputs_source, normedinputs_source, inputmeans_source, inputsigma_source, ncat_per_feature_source = prepdata(filename=rootfile_source)
    print(ncat_per_feature_target)
    inputdim = len(featurevars)-1
    ncat_per_feature_target = ncat_per_feature_target[0:inputdim]
    conddim = normedinputs_target.shape[1] - inputdim

    issignal_target = (rawinputs_target['regions']==0) # signal_selection 
    issignal_source = (rawinputs_source['regions']==0) # signal_selection 
    isbackground_target = ~issignal_target
    isbackground_source = ~issignal_source
    bkgnormed_target = normedinputs_target[isbackground_target]
    bkgnormed_source = normedinputs_source[isbackground_source]
    xmax = np.reshape(inputmeans_source + 5* inputsigma_source, inputmeans_source.shape[1])

    m = ABCDdnn(ncat_per_feature_target, inputdim, minibatch=minibatch, conddim=conddim, LRrange=LRrange, \
        beta1=beta1, beta2=beta2, nafdim=nafdim, depth=depth, savedir=savedir, retrain=retrain, seed=seed, permute=permute)
    m.setrealdata(bkgnormed_target, bkgnormed_source)
    m.savehyperparameters()
    m.monitorevery = 100

    
    condlist = {
            "CR1": [[0., 1., 0., 0., 0., 0.]],
            "CR2": [[0., 0., 1., 0., 0., 0.]],
            "CR3": [[0., 0., 0., 1., 0., 0.]],
            "CR4": [[0., 0., 0., 0., 1., 0.]],
            "CR5": [[0., 0., 0., 0., 0., 1.]]
            }
    '''
    condlist = {
            "CR1": [[0., 1., 0., 0., 0.]],
            "CR2": [[0., 0., 1., 0., 0.]],
            "CR3": [[0., 0., 0., 1., 0.]],
            "CR4": [[0., 0., 0., 0., 1.]]
            }
    '''

    if train:
        m.train(steps, condlist)
        m.display_training()
    
    ncol=4 # for plots below

    
    select_data_target = {
        "SR": (rawinputs_target['regions']==0),
        "CR1": (rawinputs_target['regions']==1),
        "CR2": (rawinputs_target['regions']==2),
        "CR3": (rawinputs_target['regions']==3),
        "CR4": (rawinputs_target['regions']==4),
        "CR5": (rawinputs_target['regions']==5)
    }

    select_data_source = {
        "SR": (rawinputs_source['regions']==0),
        "CR1": (rawinputs_source['regions']==1),
        "CR2": (rawinputs_source['regions']==2),
        "CR3": (rawinputs_source['regions']==3),
        "CR4": (rawinputs_source['regions']==4),
        "CR5": (rawinputs_source['regions']==5)
    }
    
    '''
    select_data_target = {
        "SR": (rawinputs_target['regions_merge']==0),
        "CR1": (rawinputs_target['regions_merge']==1),
        "CR2": (rawinputs_target['regions_merge']==2),
        "CR3": (rawinputs_target['regions_merge']==3),
        "CR4": (rawinputs_target['regions_merge']==4)
    }

    select_data_source = {
        "SR": (rawinputs_source['regions_merge']==0),
        "CR1": (rawinputs_source['regions_merge']==1),
        "CR2": (rawinputs_source['regions_merge']==2),
        "CR3": (rawinputs_source['regions_merge']==3),
        "CR4": (rawinputs_source['regions_merge']==4)
    }
    '''

    select_data_source_transfered = {}
    select_data_source_transfered_wonorm = {}

    runpredict = False

    if runpredict:

        plottextlist=['SR','CR1','CR2','CR3','CR4','CR5']
        #plottextlist=['SR','CR1','CR2','CR3','CR4']

        for r in plottextlist:
            select_data_source_transfered[r] = []
            select_data_source_transfered_wonorm[r] = []

            transferedlist = []

            xin = normedinputs_source[select_data_source[r]]
            xgen = m.model.predict(xin)
            transferedlist.append(xgen)
            
            transfered_data= np.vstack(transferedlist)
            transfered_data_norm = transfered_data * inputsigma_target[:, :inputdim] + inputmeans_target[:, :inputdim]
            ntransfered_data = transfered_data_norm.shape[0]

            select_data_source_transfered[r].append(transfered_data_norm)
            select_data_source_transfered_wonorm[r].append(transfered_data)

    labelsindices = [['H_mass', 'H_mass', 105., 170, 50], 
                    ['BDT_score', 'BDT_score', 0., 1., 50],
                 #['H_ptt', 'H_ptt', 0.0, 50., 50], 
                 ['H_relpt', 'H_relpt', 0.0, 3., 50], 
                 #['H_pt', 'H_pt', 0.0, 50., 50], 
                 ['Z_cos_theta', 'Z_cos_theta', -1., 1., 50], 
                 ['lep_cos_theta', 'lep_cos_theta', -1., 1., 50],
                 #['lep_phi', 'lep_phi', -3.5, 3.5, 50],
                 #['l1g_deltaR', 'l1g_deltaR', 0., 6., 50],
                 #['l2g_deltaR', 'l2g_deltaR', 0., 6., 50]
                 ['gamma_pt', 'gamma_pt', 0., 80., 50],
                 ['Z_pt', 'Z_pt', 0., 80., 50]
                 #['Z_lead_lepton_pt', 'Z_lead_lepton_pt', 0., 100., 50],
                 #['Z_sublead_lepton_pt', 'Z_sublead_lepton_pt', 0., 80., 50]
                 #['gamma_mvaID', 'gamma_mvaID', -1.,1., 50],
                 #['n_jets', 'n_jets', 0., 5., 5],
                 #['n_leptons', 'n_leptons', 0., 6., 6]
                 ]
    labelsindices_unnorm = [
                            ['H_mass', 'H_mass', -2, 4, 50], 
                            ['BDT_score', 'BDT_score', -2, 4, 50],
                            #['H_ptt', 'H_ptt', -2, 4, 50], 
                            ['H_relpt', 'H_relpt', -2, 4, 50], 
                            #['H_pt', 'H_pt', -2, 4, 50], 
                            ['Z_cos_theta', 'Z_cos_theta', -2, 4, 50], 
                            ['lep_cos_theta', 'lep_cos_theta', -2, 4, 50],
                            #['lep_phi', 'lep_phi', -2, 4, 50], 
                            #['l1g_deltaR', 'l1g_deltaR', -2, 4, 50], 
                            #['l2g_deltaR', 'l2g_deltaR', -2, 4, 50]
                            ['gamma_pt', 'gamma_pt', -2, 4, 50], 
                            ['Z_pt', 'Z_pt', -2, 4, 50]
                            #['Z_lead_lepton_pt', 'Z_lead_lepton_pt', -2, 4, 50], 
                            #'Z_sublead_lepton_pt', 'Z_sublead_lepton_pt', -2, 4, 50]
                            #['gamma_mvaID', 'gamma_mvaID', -2, 4, 50],
                            #['n_jets', 'n_jets', -2, 4, 50],
                            #['n_leptons', 'n_leptons', -2, 4, 50]
                            ]
    
    runplots = False

    if runplots:
        yscales = ['log', 'linear']
        for yscale in yscales:
            for li in labelsindices:
                pos = featurevars.index(li[1])
                nbins = li[-1]
                
                plts = {}
                plt.figure(figsize=(12, 8))

                for r in plottextlist:
                    #print(rawinputs_target[select_data_target[r]])
                    target_data = rawinputs_target[select_data_target[r]]
                    source_data = rawinputs_source[select_data_source[r]]
                    source_transfered_data = select_data_source_transfered[r][0]

                    plts[r] = plt.subplot(2, 3, plottextlist.index(r)+1)
                    plts[r].hist(source_transfered_data[:,pos], bins=nbins, alpha=0.5, range=(li[2], li[3]), histtype='bar', density=True, label='SM ZG (after morph)')

                    hist1, bins = np.histogram(target_data[li[1]],bins=nbins, range=(li[2], li[3]), density=True)
                    scale = len(target_data[li[1]]) / sum(hist1)
                    err = np.sqrt(hist1 * scale) / scale
                    center = (bins[:-1] + bins[1:]) / 2
                    plts[r].errorbar(center, hist1, yerr=err, fmt='.', c='r', markersize=8,capthick=0, label='Data')

                    plts[r].hist(source_data[li[1]], bins=nbins, alpha=0.5, range=(li[2], li[3]), histtype='step', density=True, label='SM ZG (before morph)')
                    plt.yscale(yscale)
                    plts[r].set_title(r)
                    plts[r].set_xlabel(li[1])
                    plts[r].legend(loc="upper right")
                    
                plt.tight_layout()
                plt.show()
                plt.savefig(os.path.join(savedir, f'result_matrix_{li[1]}_{yscale}.pdf'))
                plt.savefig(os.path.join(savedir, f'result_matrix_{li[1]}_{yscale}.png'))
                plt.close()

            for li in labelsindices_unnorm:
                pos = featurevars.index(li[1])
                nbins = li[-1]
                
                plts = {}
                plt.figure(figsize=(12, 8))

                for r in plottextlist:
                    #print(rawinputs_target[select_data_target[r]])
                    target_data = normedinputs_target[select_data_target[r]]
                    source_data = normedinputs_source[select_data_source[r]]
                    source_transfered_data = select_data_source_transfered_wonorm[r][0]

                    plts[r] = plt.subplot(2, 3, plottextlist.index(r)+1)
                    plts[r].hist(target_data[:,pos], bins=nbins, alpha=0.5, range=(li[2], li[3]), histtype='bar', density=True, label='target')
        
                    hist1, bins = np.histogram(source_transfered_data[:,pos],bins=nbins, range=(li[2], li[3]), density=True)
                    scale = len(source_transfered_data[:,pos]) / sum(hist1)
                    err = np.sqrt(hist1 * scale) / scale
                    center = (bins[:-1] + bins[1:]) / 2
                    plts[r].errorbar(center, hist1, yerr=err, fmt='.', c='r', markersize=8,capthick=0, label='transfermed')

                    plts[r].hist(source_data[:,pos], bins=nbins, alpha=0.5, range=(li[2], li[3]), histtype='step', density=True, label='source')
                    plt.yscale(yscale)
                    plts[r].set_title(r)
                    plts[r].set_xlabel(li[1])
                    plts[r].legend(loc="upper right")
                    
                plt.tight_layout()
                plt.show()
                plt.savefig(os.path.join(savedir, f'result_matrix_{li[1]}_{yscale}_unnorm.png'))
                plt.close()


    save_results = False
    if save_results:

        hzg = uproot.open(rootfile_source)
        #treename = "zero_to_one_jet"
        #treename = "two_jet"
        treename = "inclusive"
        hzgtree = hzg[treename]

        iscategorical = [False, False, False, False, False, False, False, True]
        _onehotencoder = OneHotEncoder_int(iscategorical)

        arrays = hzgtree.arrays(featurevars, library="pd")
        arrays_all = pd.DataFrame(hzgtree.arrays(library="pd"))
        inputtmp = pd.DataFrame(arrays)
        inputnumpy = inputtmp.to_numpy(dtype=np.float32)
        inputs = _onehotencoder.encode(inputnumpy)
        ncats = _onehotencoder.ncats
        ncat_per_feature = _onehotencoder.categories_per_feature

        data = uproot.open(rootfile_target)
        datatree = data[treename]
        arrays_data = pd.DataFrame(datatree.arrays(library="pd"))

        normedinputs_source = (inputs-inputmeans_source) / inputsigma_source

        transferedlist = []
        xin = normedinputs_source
        xgen = m.model.predict(xin)
        transferedlist.append(xgen)

        transfered_data= np.vstack(transferedlist)
        transfered_data = transfered_data * inputsigma_target[:, :inputdim] + inputmeans_target[:, :inputdim]

        BDT_filename = "/publicfs/cms/user/wangzebing/HZG/model_HZGamma_KinematicBDT_Hptt_inclusive.pkl"

        model = pickle.load(open(BDT_filename, 'rb'))
        arrays_all["BDT_score"] = model.predict_proba(arrays_all[["Z_cos_theta", "lep_cos_theta", "H_ptt", "Z_lead_lepton_eta", "Z_sublead_lepton_eta", "gamma_eta", "lep_phi", "gamma_mvaID", "gamma_ptRelErr", "l1g_deltaR", "l2g_deltaR"]].to_numpy())[:, 1]
        arrays_data["BDT_score"] = model.predict_proba(arrays_data[["Z_cos_theta", "lep_cos_theta", "H_ptt", "Z_lead_lepton_eta", "Z_sublead_lepton_eta", "gamma_eta", "lep_phi", "gamma_mvaID", "gamma_ptRelErr", "l1g_deltaR", "l2g_deltaR"]].to_numpy())[:, 1]

        for ifeature in range(inputdim):
            arrays_all[featurevars[ifeature]+'_transfered'] = np.array(transfered_data[:,ifeature])
        
        #arrays_all["BDT_score_transfered"] = model.predict_proba(arrays_all[["Z_cos_theta_transfered", "lep_cos_theta_transfered", "H_relpt_transfered", "Z_lead_lepton_eta", "Z_sublead_lepton_eta", "gamma_eta", "lep_phi", "gamma_mvaID", "gamma_ptRelErr", "l1g_deltaR", "l2g_deltaR", "gamma_relpt"]].to_numpy())[:, 1]


        arrays_all.to_root(savedir+'/ZGToLLG_{}.root'.format(treename), key=treename, mode='a', index=False)
        arrays_data.to_root(savedir+'/data_{}.root'.format(treename), key=treename, mode='a', index=False)
