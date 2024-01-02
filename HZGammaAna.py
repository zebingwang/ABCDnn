from ABCD_dnn_mmd import ABCDdnn
import numpy as np
import matplotlib.pyplot as plt
import uproot
from onehotencoder import OneHotEncoder_int
import os
import pandas as pd

#featurevars = ['met', 'ht', 'pt5', 'pt6', 'njet', 'nbtag']

#rootfile='ttjjresult.root'

featurevars = ['H_mass', 'H_pt', 'Z_pt', 'Z_eta', 'Z_phi', 'gamma_pt', 'gamma_eta', 'gamma_phi', 'gamma_mvaID', 'n_jets', 'n_leptons', 'regions']
rootfile_target='./datasets/data.root'
rootfile_source='./datasets/ZGToLLG.root'

def prepdata(filename):
    hzg = uproot.open(filename)
    hzgtree = hzg['passedEvents']
    iscategorical = [False, False, False, False, False, False, False, False, False, False, False, True]
    _onehotencoder = OneHotEncoder_int(iscategorical)

    arrays = hzgtree.arrays(featurevars, library="pd")
    inputtmp = pd.DataFrame(arrays)

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
            meanslist.append(mean)
            sigma = np.std(inputnumpy[:, currentcolumn], axis=0, dtype=np.float32).reshape(1,1)
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
    savedir='abcdnn', seed=100, retrain=False, train=True):
    #rawinputs, normedinputs, inputmeans, inputsigma, ncat_per_feature = prepdata()
    rawinputs_target, normedinputs_target, inputmeans_target, inputsigma_target, ncat_per_feature_target = prepdata(rootfile_target)
    rawinputs_source, normedinputs_source, inputmeans_source, inputsigma_source, ncat_per_feature_source = prepdata(rootfile_source)
    print(ncat_per_feature_target)
    inputdim = 11
    ncat_per_feature_target = ncat_per_feature_target[0:inputdim]
    conddim = normedinputs_target.shape[1] - inputdim

    issignal_target = (rawinputs_target['regions']==0) # signal_selection 
    issignal_source = (rawinputs_source['regions']==0) # signal_selection 
    isbackground_target = ~issignal_target
    isbackground_source = ~issignal_source
    bkgnormed_target = normedinputs_target[isbackground_target]
    bkgnormed_source = normedinputs_source[isbackground_source]
    bkg_target = rawinputs_target[isbackground_target]
    bkg_source = rawinputs_source[isbackground_source]   

    m = ABCDdnn(ncat_per_feature_target, inputdim, minibatch=minibatch, conddim=conddim, LRrange=LRrange, \
        beta1=beta1, beta2=beta2, nafdim=nafdim, depth=depth, savedir=savedir, retrain=retrain, seed=seed)
    m.setrealdata(bkgnormed_target, bkgnormed_source)
    m.savehyperparameters()
    m.monitorevery = 100

    if train:
        m.train(steps)
        m.display_training()

    
    ncol=4 # for plots below
    condlist = {
                "SR": [[1., 0., 0., 0., ]],
                "CR1": [[0., 1., 0., 0., ]],
                "CR2": [[0., 0., 1., 0., ]],
                "CR3": [[0., 0., 0., 1., ]]
                }

    select_data_target = {
        "SR": (rawinputs_target['regions']==0),
        "CR1": (rawinputs_target['regions']==1),
        "CR2": (rawinputs_target['regions']==2),
        "CR3": (rawinputs_target['regions']==3)
    }

    select_data_source = {
        "SR": (rawinputs_source['regions']==0),
        "CR1": (rawinputs_source['regions']==1),
        "CR2": (rawinputs_source['regions']==2),
        "CR3": (rawinputs_source['regions']==3)
    }

    select_data_source_transfered = {}

    plottextlist=['SR','CR1','CR2','CR3']

    for r in plottextlist:
        select_data_source_transfered[r] = []

        transferedlist = []

        xin = normedinputs_source[select_data_source[r]]
        xgen = m.model.predict(xin)
        transferedlist.append(xgen)
        
        transfered_data= np.vstack(transferedlist)
        transfered_data = transfered_data * inputsigma_source[:, :inputdim] + inputmeans_source[:, :inputdim]
        ntransfered_data = transfered_data.shape[0]

        select_data_source_transfered[r].append(transfered_data)

    labelsindices = [['H_mass', 'H_mass', 105., 170, 100], 
                 ['H_pt', 'H_pt', 0.0, 100., 100],
                 ['Z_pt', 'Z_pt', 0.0, 100., 100], 
                 ['Z_eta', 'Z_eta', -4., 4., 50],
                 ['Z_phi', 'Z_phi', -4., 4., 50],
                 ['gamma_pt', 'gamma_pt', 0., 50., 100],
                 ['gamma_eta', 'gamma_eta', -4., 4., 50],
                 ['gamma_phi', 'gamma_phi', -4., 4., 50],
                 ['gamma_mvaID', 'gamma_mvaID', 0.,1., 50],
                 ['n_jets', 'n_jets', 0., 5., 5],
                 ['n_leptons', 'n_leptons', 0., 6., 6]]
    runplots = True

    if runplots:
        yscales = ['log', 'linear']
        for yscale in yscales:
            for li in labelsindices:
                pos = featurevars.index(li[1])
                nbins = li[-1]
                
                for r in plottextlist:
                    #print(rawinputs_target[select_data_target[r]])
                    target_data = rawinputs_target[select_data_target[r]]
                    source_data = rawinputs_source[select_data_source[r]]
                    source_transfered_data = select_data_source_transfered[r][0]

                    #plt.figure(figsize=(6, 3))
                    plt.subplot(2, 2, plottextlist.index(r)+1)
                    #ax[row,col].set_xlabel(f"${li[0]}$")
                    plt.hist(target_data[li[1]], bins=nbins, alpha=0.5, range=(li[2], li[3]), histtype='bar', density=True, label='target')
        
                    hist1, bins = np.histogram(source_transfered_data[:,pos],bins=nbins, range=(li[2], li[3]), density=True)
                    scale = len(source_transfered_data[:,pos]) / sum(hist1)
                    err = np.sqrt(hist1 * scale) / scale
                    center = (bins[:-1] + bins[1:]) / 2
                    plt.errorbar(center, hist1, yerr=err, fmt='.', c='r', markersize=8,capthick=0, label='transfermed')

                    plt.hist(source_data[li[1]], bins=nbins, alpha=0.5, range=(li[2], li[3]), histtype='step', density=True, label='source')
                    plt.yscale(yscale)
                    plt.title(r)
                    plt.xlabel(li[1])
                    plt.legend(loc="upper right")
                    
                plt.tight_layout(pad=1.5)
                plt.savefig(os.path.join(savedir, f'result_matrix_{li[1]}_{yscale}.pdf'))
                plt.savefig(os.path.join(savedir, f'result_matrix_{li[1]}_{yscale}.png'))
                plt.show()