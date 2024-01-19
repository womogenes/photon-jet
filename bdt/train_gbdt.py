import sys
sys.path.append("..")

from utils import data_dir

import os
import random
#import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
parameters = {'axes.labelsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'legend.fontsize': 12,
          'lines.linewidth' : 2,
          'lines.markersize' : 7}
plt.rcParams.update(parameters)



import ROOT 
import joblib
import ctypes

import seaborn as sns
import pandas as pd
import pandas.core.common as com
#from pandas.core.index import Index

from pandas.plotting import scatter_matrix
#import imblearn
#from imblearn.combine import SMOTETomek
#from imblearn.over_sampling import SMOTE

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
#from sklearn.learning_curve import learning_curve
#from sklearn import grid_search


ROOT.gStyle.SetPadLeftMargin(0.16);
ROOT.gStyle.SetPadRightMargin(0.04);
ROOT.gStyle.SetPadTopMargin(0.1);
ROOT.gStyle.SetPadBottomMargin(0.15);
ROOT.gStyle.SetTitleSize(0.04, "xy");
ROOT.gStyle.SetLabelSize(0.04, "xy");
ROOT.gStyle.SetTitleOffset(1.2,"x");
ROOT.gStyle.SetTitleOffset(1.2, "y");
ROOT.gStyle.SetNdivisions(505, "y");

def usage():
	print ('test usage')
	sys.stdout.write('''
			SYNOPSIS
			./BDT_pre.py tr_sig tr_bkg0 tr_bkg1 test_sig test_bkg0 test_bkg1 
			AUTHOR
			Yanxi Gu <GUYANXI@ustc.edu>
			DATE
			10 Jan 2021
			\n''')


def findBin(bins, var, ibin):
    for i in range(len(bins)-1):
        if var>=bins[i] and var<bins[i+1]:
           ibin.append(i)



def main():
    args = sys.argv[1:]
    if len(args) < 3:
        return usage()

    print ('part1')   

    # get root files and convert them to array
    branch_names = """frac_first,first_lateral_width_eta_w20,first_lateral_width_eta_w3,first_fraction_fside,first_dEs,first_Eratio,second_R_eta,second_R_phi,second_lateral_width_eta_weta2""".split(",")
    branch_labels = {"frac_first" : "$f_{1}$",
            "first_lateral_width_eta_w20" : "$w_{s20}$",
            "first_lateral_width_eta_w3" : "$w_{s3}$",
            "first_fraction_fside": "$f_{side}$",
            "first_dEs" : "$\Delta E_{s}$",
            "first_Eratio" : "$E_{ratio}$",
            "second_R_eta": "$R_{\eta}$",
            "second_R_phi": "$R_{\phi}$",
            "second_lateral_width_eta_weta2": "$w_{\eta2}$"
            }
    branch_ene = """total_e""".split(",")

    ## WILLIAM'S ADDITION
    task = "scalar1"
    
    tag = args[0]
    fin1 = ROOT.TFile(args[1])
    fin2 = ROOT.TFile(args[2])
    fin3 = ROOT.TFile(args[3])
    fin4 = ROOT.TFile(args[4])
    fin5 = ROOT.TFile(args[5])
    fin6 = ROOT.TFile(args[6])

    output = args[7]
    binned = True 
    #print("binned ", binned)

    sigLegend = "";
    bg0Legend = "$\gamma$"
    bg1Legend = "$\pi^{0}$"
    if "axion1" in output:
        sigLegend = "$a \\rightarrow \gamma\gamma$"
    elif "axion2" in output:
        #sigLegend = "$a \\rightarrow 3\pi^{0} \\rightarrow 6\gamma$"
        sigLegend = "$a \\rightarrow 3\pi^{0}$"
    elif "scalar1" in output:
        #sigLegend = "$a \\rightarrow \pi^{0}\pi^{0} \\rightarrow 4\gamma$"
        sigLegend = "$h_2 \\rightarrow \pi^{0}\pi^{0}$"


    # ########### Train samples #############
    train_nEvents = 70000
    train_tree1 = fin1.Get("fancy_tree")
    train_sig0 = train_tree1.AsMatrix(columns=branch_names)
    train_sig0_ene = train_tree1.AsMatrix(columns=branch_ene)
    train_signal0 = train_sig0[:np.int(train_nEvents),:]
    train_signal0_ene = train_sig0_ene[:np.int(train_nEvents),:]
    
    train_tree2 = fin2.Get("fancy_tree")
    train_backgr0 = train_tree2.AsMatrix(columns=branch_names)
    train_backgr0_ene = train_tree2.AsMatrix(columns=branch_ene)
    train_background0 = train_backgr0[:np.int(train_nEvents),:]
    train_background0_ene = train_backgr0_ene[:np.int(train_nEvents),:]
    
    train_tree3 = fin3.Get("fancy_tree")
    train_backgr1 = train_tree3.AsMatrix(columns=branch_names)
    train_backgr1_ene = train_tree3.AsMatrix(columns=branch_ene)
    train_background1 = train_backgr1[:np.int(train_nEvents),:]
    train_background1_ene = train_backgr1_ene[:np.int(train_nEvents),:]

    # ########### Test samples #############
    test_nEvents = 30000
    test_tree1 = fin4.Get("fancy_tree")
    test_sig0 = test_tree1.AsMatrix(columns=branch_names)
    test_sig0_ene = test_tree1.AsMatrix(columns=branch_ene)
    test_signal0 = test_sig0[:np.int(test_nEvents),:]
    test_signal0_ene = test_sig0_ene[:np.int(test_nEvents),:]

    test_tree3 = fin5.Get("fancy_tree")
    test_backgr0 = test_tree3.AsMatrix(columns=branch_names)
    test_backgr0_ene = test_tree3.AsMatrix(columns=branch_ene)
    test_background0 = test_backgr0[:np.int(test_nEvents),:]
    test_background0_ene = test_backgr0_ene[:np.int(test_nEvents),:]

    test_tree4 = fin6.Get("fancy_tree")
    test_backgr1 = test_tree4.AsMatrix(columns=branch_names)
    test_backgr1_ene = test_tree4.AsMatrix(columns=branch_ene)
    test_background1 = test_backgr1[:np.int(test_nEvents),:]
    test_background1_ene = test_backgr1_ene[:np.int(test_nEvents),:]


    #==================================
    plot_inputs("./gbdt_results_9var/" + tag + "/" + output+"/", branch_names, branch_labels, train_signal0, None, train_background0, None, train_background1,None, sigLegend, bg0Legend, bg1Legend)
    return
    plot_correlations("./gbdt_results_9var/" + tag + "/" + output+"/", branch_names, branch_labels, train_signal0, train_background0, train_background1)
    #==================================
    

    # for sklearn data is usually organised into one 2D array of shape (n_samples * n_features)
    # containing all the data and one array of categories of length n_samples
    train_X_raw = np.concatenate((train_signal0, train_background0, train_background1))
    train_X_raw_ene = np.concatenate((train_signal0_ene, train_background0_ene, train_background1_ene))
    test_X_raw = np.concatenate((test_signal0, test_background0, test_background1))
    test_X_raw_ene = np.concatenate((test_signal0_ene, test_background0_ene, test_background1_ene))
 
    #These are set manually
    #======================= 
    processLabels = {sigLegend:1, bg0Legend:0, bg1Legend:2}
    processColumns = [bg0Legend, sigLegend, bg1Legend] 
    iColForSig = 1 
    #======================= 

    sortedLabels = [] 
    for key in processLabels:
        sortedLabels.append(processLabels[key])
    sortedLabels.sort()

    train_y_raw = np.concatenate((np.zeros(train_signal0.shape[0])+processLabels[sigLegend], np.zeros(train_background0.shape[0])+processLabels[bg0Legend], np.zeros(train_background1.shape[0])+processLabels[bg1Legend]))
    test_y_raw = np.concatenate((np.zeros(test_signal0.shape[0])+processLabels[sigLegend], np.zeros(test_background0.shape[0])+processLabels[bg0Legend], np.zeros(test_background1.shape[0])+processLabels[bg1Legend]))
    
    print(len(train_signal0))
    print(len(test_signal0))

    print ('part2')
    for key in processLabels:
        print("Length for ", key, "is ", len(test_y_raw[test_y_raw==processLabels[key]]))


    """
    Training Part
    """
    # Train and test
    #X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.30, random_state=3443)
    X_train = train_X_raw
    #https://datascience.stackexchange.com/questions/11928/valueerror-input-contains-nan-infinity-or-a-value-too-large-for-dtypefloat32 
    print("check X_train NaN") 
    np.where(np.isnan(X_train)) 
    print("check X_train Inf") 
    np.where(np.isinf(X_train)) 
    #print("Replace X_train") 
    #X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    #X_train = np.nan_to_num(X_train.astype(np.float32))


    X_test = test_X_raw
    X_test_ene = test_X_raw_ene
    #X_test_comb = list(zip(X_test, X_test_ene))
    #print("X_test_comb", X_test_comb)
    
    y_train = train_y_raw
    y_test = test_y_raw

    ###################################################################

    #category the X_test into a few sub arrays
    #ene_bins = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 250]
    ene_bins = [0, 100, 150, 200, 300]
    #sigEff = ROOT.TEfficiency("Signal efficiency", "", len(ene_bins)-1, ctypes.c_void_p(ene_bins.ctypes.data)) 
    #bkg0Eff = ROOT.TEfficiency("Bkg0 efficiency", "", len(ene_bins)-1, ctypes.c_void_p(ene_bins.ctypes.data)) 
    #bkg1Eff = ROOT.TEfficiency("Bkg1 efficiency", "", len(ene_bins)-1, ctypes.c_void_p(ene_bins.ctypes.data)) 
    #sigEff = ROOT.TEfficiency("Signal efficiency", "", 12, 19, 271) 
    #bkg0Eff = ROOT.TEfficiency("Bkg0 efficiency", "", 12, 19, 271) 
    #bkg1Eff = ROOT.TEfficiency("Bkg1 efficiency", "", 12, 19, 271) 
    sigEff = ROOT.TEfficiency("Signal efficiency", "", 10, 40, 250) 
    bkg0Eff = ROOT.TEfficiency("Bkg0 efficiency", "", 10, 40, 250) 
    bkg1Eff = ROOT.TEfficiency("Bkg1 efficiency", "", 10, 40, 250) 
    sigEff_binned = ROOT.TEfficiency("Binned signal efficiency", "", 10, 40, 250) 
    bkg0Eff_binned = ROOT.TEfficiency("Binned bkg0 efficiency", "", 10, 40, 250) 
    bkg1Eff_binned = ROOT.TEfficiency("Binned bkg1 efficiency", "", 10, 40, 250) 
    
    sigEff.SetTitle("%s;E_{a} [GeV];Efficiency"%("Unbinned training")); 
    bkg0Eff.SetTitle(";E_{a} [GeV];"); 
    
    sigEff_binned.SetTitle("%s;E_{a} [GeV];Efficiency"%("Binned training")); 
    bkg0Eff_binned.SetTitle(";E_{a} [GeV];"); 

    ene = []
    for i in range(len(ene_bins)-1):
        ene.append((ene_bins[i]+ene_bins[i+1])/2)

    ###################################################################
    #The training is done in 10 different bins
#    X_train_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_train_sig_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_train_bkg0_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_train_bkg1_binned = [[], [], [], [], [], [], [], [], [], []]
#    y_train_binned = [[], [], [], [], [], [], [], [], [], []]
#   
#    X_test_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_test_sig_binned = [[], [], [], [], [], [], [], [], [], []]
#    y_test_binned = [[], [], [], [], [], [], [], [], [], []]
#    
#    X_test_bkg0_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_test_bkg1_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_test_ene_binned = [[], [], [], [], [], [], [], [], [], []]
#    
#    X_test_sig_ene_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_test_bkg0_ene_binned = [[], [], [], [], [], [], [], [], [], []]
#    X_test_bkg1_ene_binned = [[], [], [], [], [], [], [], [], [], []]

    #The training is done in 4 different bins
    X_train_binned = [[], [], [], []]
    X_train_sig_binned = [[], [], [], []]
    X_train_bkg0_binned = [[], [], [], []]
    X_train_bkg1_binned = [[], [], [], []]
    y_train_binned = [[], [], [], []] 

    X_test_binned = [[], [], [], []] 
    X_test_sig_binned = [[], [], [], []] 
    y_test_binned = [[], [], [], []] 

    X_test_bkg0_binned = [[], [], [], []] 
    X_test_bkg1_binned = [[], [], [], []] 
    X_test_ene_binned = [[], [], [], []] 

    X_test_sig_ene_binned = [[], [], [], []] 
    X_test_bkg0_ene_binned = [[], [], [], []] 
    X_test_bkg1_ene_binned = [[], [], [], []] 


    # Categorize the train_X
    for i in range (len(train_X_raw)):
        ibin=[]
        #Find which bin this belongs
        findBin(ene_bins, train_X_raw_ene[i][0]/1000., ibin)
        if len(ibin)>0:
            y_train_binned[ibin[0]].append(train_y_raw[i])

            X_raw_col = []
            for j in range (len(train_X_raw[i])):
                X_raw_col.append(train_X_raw[i][j])

            X_train_binned[ibin[0]].append(X_raw_col)
            #print("X_raw_col = ", X_raw_col, "label", train_y_raw[i]) 
            if train_y_raw[i] ==processLabels[sigLegend] :
                X_train_sig_binned[ibin[0]].append(X_raw_col)
            elif train_y_raw[i] == processLabels[bg0Legend]:
                X_train_bkg0_binned[ibin[0]].append(X_raw_col)
            elif train_y_raw[i] == processLabels[bg1Legend]:
                X_train_bkg1_binned[ibin[0]].append(X_raw_col)

    # Categorize the test_X
    for i in range (len(test_X_raw)):
        ibin=[]
        findBin(ene_bins, test_X_raw_ene[i][0]/1000., ibin)
        if len(ibin)>0:
            y_test_binned[ibin[0]].append(test_y_raw[i])
           
            #Copy
            X_raw_col = []
            for j in range (len(test_X_raw[i])):
                X_raw_col.append(test_X_raw[i][j])
            X_raw_ene_col = []
            for j in range (len(test_X_raw_ene[i])):
                X_raw_ene_col.append(test_X_raw_ene[i][j])

            X_test_binned[ibin[0]].append(X_raw_col)
            X_test_ene_binned[ibin[0]].append(X_raw_ene_col)
           
            if test_y_raw[i] == processLabels[sigLegend] :
                X_test_sig_binned[ibin[0]].append(X_raw_col)
                X_test_sig_ene_binned[ibin[0]].append(X_raw_ene_col)
            elif test_y_raw[i] == processLabels[bg0Legend]:
                X_test_bkg0_binned[ibin[0]].append(X_raw_col)
                X_test_bkg0_ene_binned[ibin[0]].append(X_raw_ene_col)
            elif test_y_raw[i] == processLabels[bg1Legend]:
                X_test_bkg1_binned[ibin[0]].append(X_raw_col)
                X_test_bkg1_ene_binned[ibin[0]].append(X_raw_ene_col)


    ###################################################################

    print ('training')   
    bdt = GradientBoostingClassifier(max_depth=5, min_samples_leaf=200, min_samples_split=10,  n_estimators=100, learning_rate=1.0)
    bdt.fit(X_train, y_train)

    print("Accuracy score (training): {0:.3f}".format(bdt.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(bdt.score(X_test, y_test)))

    print ('Save the importance')   
    importances = bdt.feature_importances_
    f = open('gbdt_results_9var/'+ tag + '/' + output+'/output_importance.txt', 'w')
    f.write("%-25s%-15s\n"%('Variable Name','Output Importance'))
    for i in range (len(branch_names)):
        f.write("%-25s%-15s\n"%(branch_names[i], importances[i]))
        print("%-25s%-15s\n"%(branch_names[i], importances[i]), file=f)
    f.close() 

    #y_predicted = bdt.predict(X_train)
    y_predicted = bdt.predict(X_test)

    # IMPORTANT: The dimension of decisions_train is equal to the number of labels (one-vs-rest for each label)
    # The colomn i is the score for distinguishing the i-th label vs. others:
    # the labels are ordered in increasing order, for examples:
    # - if Label_bg0=0, Label_bg1=2, Label_sig=1, the second colum is for signal
    # - if Label_bg0=0, Label_bg1=2, Label_sig=3, the third colum is for signal
    decisions_train = bdt.decision_function(X_train)
    decisions_test = bdt.decision_function(X_test)
    #print("decisions 1 size ", decisions_train.shape[1]) 


    filepath = 'ROC'
    joblib.dump(bdt, './gbdt_results_9var/' + tag + '/' + output+'/'+filepath+'/bdt_model.pkl')
    compare_train_test(bdt, processLabels, iColForSig, X_train, y_train, X_test, y_test, "./gbdt_results_9var/" + tag + "/" + output, filepath, "", 50, sigLegend, bg0Legend, bg1Legend)


    # The numbers should always be in increasing order to be consistent with what's stored in the decisions
    # eg. classes = [2,3,4] indicating the first/second/third cololum has label= 2,3,4 or not
    y_train = label_binarize(y_train, classes = [sortedLabels[0], sortedLabels[1], sortedLabels[2]])
    y_test = label_binarize(y_test, classes = [sortedLabels[0], sortedLabels[1], sortedLabels[2]])
    y_predicted = label_binarize(y_predicted, classes = [sortedLabels[0], sortedLabels[1], sortedLabels[2]])
    n_classes = y_test.shape[1] 
    print("n_classes=", n_classes)

    np.set_printoptions(precision=2)
    # Print confusion matrix
    for i in range(n_classes):
        others = "Backgrounds"
        
        print("Confusion Matrix for :", processColumns[i])
        #cm = confusion_matrix(y_predicted[:, i], y_test[:, i], normalize = 'true')
        cm = confusion_matrix(y_test[:, i], y_predicted[:, i], normalize = 'true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[processColumns[i], others])
        disp.plot(cmap="Blues", values_format='.2f')
        plt.yticks(rotation=90)
        plt.tight_layout()
        plt.title('Confusion matrix {}-vs-others'.format(processColumns[i]))
        plt.savefig('./gbdt_results_9var/' + tag + '/' + output+'/'+filepath+'/ConfusionMatrix_%i.pdf'%(i))


    # Compute ROC curve and area under the curve for each label (one-vs-others)
    for i in range(n_classes):
        #print("label for ", i, ": ", processColumns[i])
        fpr1, tpr1, thresholds_train = roc_curve(y_train[:, i], decisions_train[:, i])
        fpr2, tpr2, thresholds_test = roc_curve(y_test[:, i], decisions_test[:, i])
        #print("fpr2 =", fpr2, "tpr2 =", tpr2, "thresholds_test =", thresholds_test)
        roc_auc1 = auc(fpr1, tpr1)
        roc_auc2 = auc(fpr2, tpr2)
        fig=plt.figure(figsize=(8,6))
        fig.patch.set_color('white')
        plt.plot(fpr1, tpr1, lw=1.2, label='train:ROC (area = %0.4f)'%(roc_auc1), color="r")
        plt.plot(fpr2, tpr2, lw=1.2, label='test: ROC (area = %0.4f)'%(roc_auc2), color="b")
        plt.plot([0,1], [0,1], '--', color=(0.6, 0.6, 0.6), label = 'Luck')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic {}-vs-others'.format(processColumns[i]))
        plt.legend(loc = "lower right")
        plt.grid()
        plt.savefig('./gbdt_results_9var/' + tag + '/' + output+'/'+filepath+'/ROC_%i.pdf'%(i))


    # Compute recall (only for the test sample here)
    # The i =0, means the sample for label equal to 1 as above 
    for i in range(n_classes):
        precision, recall, th = precision_recall_curve(y_test[:, i],
                                                       y_predicted[:, i])
        #print("th dim =", th.ndim)
        fig=plt.figure(figsize=(8,6))
        fig.patch.set_color('white')
        plt.plot(th, precision[1:], label="Precision",linewidth=5)
        plt.plot(th, recall[1:], label="Recall",linewidth=5)
        plt.title('Precision and recall for different threshold values for {}'.format(processColumns[i]))
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.legend()
        plt.grid()
        plt.savefig('./gbdt_results_9var/' + tag + '/' + output+'/'+filepath+'/PRC_%i.pdf'%(i))


    # plot efficiency

    threshold=0
    #====================================================================================
    #binned efficiency 
    #====================================================================================
    # Loop over the energy bins
    for i in range (len(X_train_binned)):
        ########################Train for each energy bin############################
        print("bin ", i, " ene = ", ene[i])
        bdt_ = GradientBoostingClassifier(max_depth=5, min_samples_leaf=200, min_samples_split=10,  n_estimators=100, learning_rate=1.0)
        #print("X_train_binned[i] = ", X_train_binned[i])
        #print("y_train_binned[i]", y_train_binned[i])
        bdt_.fit(X_train_binned[i], y_train_binned[i])  

        print("Accuracy score (training): {0:.3f}".format(bdt_.score(X_train_binned[i], y_train_binned[i])))
        print("Accuracy score (test): {0:.3f}".format(bdt_.score(X_test_binned[i], y_test_binned[i])))
        compare_train_test(bdt_, processLabels, iColForSig, np.array(X_train_binned[i]), np.array(y_train_binned[i]), np.array(X_test_binned[i]), np.array(y_test_binned[i]), "./gbdt_results_9var/" + tag + "/" + output, filepath, "_bin%i"%(i), 50, sigLegend, bg0Legend, bg1Legend)
#       #y_testbinned[:, 1] can tell us it's signal or not 
#       y_testbinned = label_binarize(y_test_binned[i], classes = [0, 1, 2])
#       decision_binned = bdt.decision_function(X_test_binned[i])
#       fpr, tpr, thresholds = roc_curve(y_testbinned[:, 1], decision_binned[:, 1])
#       for ii in range(len(thresholds)-1):
#           if thresholds[ii]>-0.1 and thresholds[ii+1]<=-0.1 :
#               print("th = ", thresholds[ii], thresholds[ii+1], "eff = ", tpr[ii+1], "fake = ", fpr[ii+1]) 
#
        #The samples for the sig, bkg0, bkg1
        decision_sig = bdt_.decision_function(X_test_sig_binned[i])
        decision_bkg0 = bdt_.decision_function(X_test_bkg0_binned[i])
        decision_bkg1 = bdt_.decision_function(X_test_bkg1_binned[i])
#       print("decision_sig size ", len(decision_sig))
#       print("decision_bkg0 size ", len(decision_bkg0))
#       print("decision_bkg1 size ", len(decision_bkg1))
#       print("decision_sig ", decision_sig)
#       print("decision_bkg0 size ", len(decision_bkg0))
#       print("decision_bkg1 size ", len(decision_bkg1))
#
#       # 1 denotes whether it's signal
        dsig_ = decision_sig[:, iColForSig]
        dbkg0_ = decision_bkg0[:, iColForSig] 
        dbkg1_ = decision_bkg1[:, iColForSig] 
            
        nSigAll = 0 
        nBkg0All = 0 
        nBkg1All = 0 
        nSigPassed = 0 
        nBkg0Passed = 0 
        nBkg1Passed = 0 
        for j in range(len(dsig_)):
            nSigAll +=1
            if dsig_[j] >= threshold :
                sigEff_binned.Fill(True, X_test_sig_ene_binned[i][j][0]/1000.)
                nSigPassed +=1
            else:
                sigEff_binned.Fill(False, X_test_sig_ene_binned[i][j][0]/1000.)
#                
        for j in range(len(dbkg0_)):
            nBkg0All +=1
            if dbkg0_[j] >= threshold :
                bkg0Eff_binned.Fill(True, X_test_bkg0_ene_binned[i][j][0]/1000.)
                nBkg0Passed +=1
            else:
                bkg0Eff_binned.Fill(False, X_test_bkg0_ene_binned[i][j][0]/1000.)
#
        for j in range(len(dbkg1_)):
            nBkg1All +=1
            if dbkg1_[j] >= threshold :
                bkg1Eff_binned.Fill(True, X_test_bkg1_ene_binned[i][j][0]/1000.)
                nBkg1Passed +=1
            else:
                bkg1Eff_binned.Fill(False, X_test_bkg1_ene_binned[i][j][0]/1000.)

        print("(nSigAll, nBkg0All, nBkg1All) = ", nSigAll, nBkg0All, nBkg1All)
        print("(nSigPassed, nBkg0Passed, nBkg1Passed) = ", nSigPassed, nBkg0Passed, nBkg1Passed)

    #====================================================================================
    # unbinned 
    #====================================================================================
    #Get the score for test sig, bkg0 and bkg1 samples, respectively
    decision_signal0 = bdt.decision_function(test_signal0)
    decision_background0 = bdt.decision_function(test_background0)
    decision_background1 = bdt.decision_function(test_background1)
    dsig = decision_signal0[:, iColForSig]
    dbkg0 = decision_background0[:, iColForSig]
    dbkg1 = decision_background1[:, iColForSig]
    for j in range(len(dsig)):
        if dsig[j] >= threshold :
            sigEff.Fill(True, test_signal0_ene[j][0]/1000.)
        else:
            sigEff.Fill(False, test_signal0_ene[j][0]/1000.)
    for j in range(len(dbkg0)):
        if dbkg0[j] >= threshold :
            bkg0Eff.Fill(True, test_background0_ene[j][0]/1000.)
        else:
            bkg0Eff.Fill(False, test_background0_ene[j][0]/1000.)
    for j in range(len(dbkg1)):
        if dbkg1[j] >= threshold :
            bkg1Eff.Fill(True, test_background1_ene[j][0]/1000.)
        else:
            bkg1Eff.Fill(False, test_background1_ene[j][0]/1000.)
 
    
    sigLabel= ""; 
    if "axion1" in output:
        sigLabel = "axion1"
    elif "axion2" in output: 
        sigLabel = "axion2"
    elif "scalar1" in output: 
        sigLabel = "scalar1"
    

    path = "./gbdt_results_9var/" + tag + "/" +output+"/"+filepath+"/";
    effname_binned = "eff_%ibins"%(len(ene_bins)-1)
    
    plot_eff(output, sigEff, bkg0Eff, bkg1Eff, path, "eff")
    plot_eff(output, sigEff_binned, bkg0Eff_binned, bkg1Eff_binned, path, effname_binned);
    save_eff(sigEff, bkg0Eff, bkg1Eff, path+ "eff_" + output+ "_" + sigLabel+".txt", path+ "eff_" + output + "_gamma.txt", path+ "eff_" + output + "_pi0.txt")
    save_eff(sigEff_binned, bkg0Eff_binned, bkg1Eff_binned, path+ effname_binned + output+ "_" + sigLabel+".txt", path+ effname_binned + output + "_gamma.txt", path+ effname_binned + output + "_pi0.txt")

    #fig=plt.figure(figsize=(8,6))
    #fig.patch.set_color('white')
    #plt.plot(ene, sigEff, lw=1.2, label='Signal', color="r")
    #plt.plot(ene, bkg0Eff, lw=1.2, label='Bkg0', color="b")
    #plt.plot(ene, bkg1Eff, lw=1.2, label='Bkg1', color="b")
    #plt.xlabel('E_{T} [GeV]')
    #plt.ylabel('Efficiency')
    #plt.legend(loc = "lower left")
    #plt.grid()


def plot_eff(output, sigEff, bkg0Eff, bkg1Eff, savepath, filename, twopanels=True):
    if twopanels:
        ROOT.gStyle.SetTitleSize(0.08, "xy");
        ROOT.gStyle.SetLabelSize(0.08, "xy");
        ROOT.gStyle.SetTitleOffset(1.1,"x");
        ROOT.gStyle.SetTitleOffset(0.8, "y");

    sigEff.SetFillStyle(3004);
    sigEff.SetFillColor(ROOT.kRed);
    sigEff.SetMarkerColor(ROOT.kRed);
    sigEff.SetLineColor(ROOT.kRed);
    sigEff.SetMarkerStyle(20);

    bkg0Eff.SetFillStyle(3005);
    bkg0Eff.SetFillColor(ROOT.kBlue);
    bkg0Eff.SetMarkerColor(ROOT.kBlue);
    bkg0Eff.SetLineColor(ROOT.kBlue);
    bkg0Eff.SetMarkerStyle(20);

    bkg1Eff.SetFillStyle(3005);
    bkg1Eff.SetFillColor(ROOT.kGreen);
    bkg1Eff.SetMarkerColor(ROOT.kGreen);
    bkg1Eff.SetLineColor(ROOT.kGreen);
    bkg1Eff.SetMarkerStyle(20);

     
    canvas = ROOT.TCanvas(filename, "", 700, 600)
    tpad = ROOT.TPad(filename+"tpad", "", 0, 0.5, 1, 1.0) 
    bpad = ROOT.TPad(filename+"bpad", "", 0, 0.05, 1, 0.5) 
    tpad.SetTopMargin(0.1);
    tpad.SetBottomMargin(0.00);
    bpad.SetTopMargin(0.0);
    bpad.SetBottomMargin(0.2);
 
    canvas.SetFillStyle(1001);
    canvas.cd()
    tpad.Draw() 
    bpad.Draw() 
    
    #sigEff.Draw("A4") 
    #bkg0Eff.Draw("same4") 
    #bkg1Eff.Draw("same4")
    
    tpad.cd() 
    sigEff.Draw("")
    ROOT.gPad.Update();
    graph = sigEff.GetPaintedGraph();
    graph.SetMinimum(0.89);
    graph.SetMaximum(1.15);
   
    x0=0.7
    x1=0.95
    y0=0.75
    y1=0.9
    if twopanels:
        x0=0.6
        x1=0.95
        y0=0.55
        y1=0.9
    leg = ROOT.TLegend(x0, y0, x1, y1)

    if "axion1" in output:
        leg.AddEntry(sigEff, "a#rightarrow #gamma #gamma","APL");
    elif "axion2" in output:
        leg.AddEntry(sigEff, "a#rightarrow 3#pi^{0} #rightarrow 6#gamma","APL");
    elif "scalar1" in output:
        leg.AddEntry(sigEff, "s#rightarrow #pi^{0}#pi^{0} #rightarrow 4#gamma","APL");
    leg.AddEntry(bkg0Eff, "#gamma","APL");
    leg.AddEntry(bkg1Eff, "#pi^{0}","APL");
    leg.SetLineStyle(0);
    leg.SetBorderSize(0);
    leg.SetFillStyle(0);
    leg.Draw();

    
    bpad.cd() 
    bkg0Eff.Draw("")
    bkg1Eff.Draw("same")
    ROOT.gPad.Update();
    graph = bkg0Eff.GetPaintedGraph();
    graph.SetMinimum(0);
    graph.SetMaximum(0.0599); #1.3
    bpad.Update();
    #ROOT.gPad.SetLogy();

    canvas.Update()
    canvas.Show()

    canvas.SaveAs("{}{}.pdf".format(savepath,filename))



def save_eff(sigEff, bkg0Eff, bkg1Eff, sigEff_file, bkg0Eff_file, bkg1Eff_file, nbins  = 10, lowEdge = 40, binwidth=21):
    print("The efficiency has ", nbins, " bins")
    
    fsig = open(sigEff_file, 'w')
    fbkg0 = open(bkg0Eff_file, 'w')
    fbkg1 = open(bkg1Eff_file, 'w')
    fsig.write('EnergyRangeLow, EnergyRangeUp, Eff, EffErrLow, EffErrUp\n') 
    fbkg0.write('EnergyRangeLow, EnergyRangeUp, Eff, EffErrLow, EffErrUp\n') 
    fbkg1.write('EnergyRangeLow, EnergyRangeUp, Eff, EffErrLow, EffErrUp\n') 
    for i in range(nbins):
        eneLow = lowEdge + i*binwidth 
        eneUp = eneLow + binwidth 
        sigEffNom = sigEff.GetEfficiency(i+1)
        sigEffErrLow = sigEff.GetEfficiencyErrorLow(i+1)
        sigEffErrUp = sigEff.GetEfficiencyErrorUp(i+1)
        
        bkg0EffNom = bkg0Eff.GetEfficiency(i+1)
        bkg0EffErrLow = bkg0Eff.GetEfficiencyErrorLow(i+1)
        bkg0EffErrUp = bkg0Eff.GetEfficiencyErrorUp(i+1)

        bkg1EffNom = bkg1Eff.GetEfficiency(i+1)
        bkg1EffErrLow = bkg1Eff.GetEfficiencyErrorLow(i+1)
        bkg1EffErrUp = bkg1Eff.GetEfficiencyErrorUp(i+1)

        fsig.write("{:.1f}, {:.1f}, {:.3f}, {:.3f}, {:.3f}\n".format(eneLow, eneUp, sigEffNom, sigEffErrLow, sigEffErrUp)) 
        fbkg0.write("{:.1f}, {:.1f}, {:.3f}, {:.3f}, {:.3f}\n".format(eneLow, eneUp, bkg0EffNom, bkg0EffErrLow, bkg0EffErrUp)) 
        fbkg1.write("{:.1f}, {:.1f}, {:.3f}, {:.3f}, {:.3f}\n".format(eneLow, eneUp, bkg1EffNom, bkg1EffErrLow, bkg1EffErrUp)) 



# Comparing train and test results for signal vs. rest
def compare_train_test(clf, processLabels, iColForSig, X_train, y_train, X_test, y_test, output, savepath, label="", bins=50, sigLegend = "Signal", bg0Legend = "Background0", bg1Legend = "Background1"):

    decisions = []
    print("label = ", label)
    for X,y in ((X_train, y_train), (X_test, y_test)):
        print("X=", X)
        print("y=", y)
        #d1 = clf.decision_function(X[y==1]).ravel()
        #d2 = clf.decision_function(X[y<0.5]).ravel()
        #d3 = clf.decision_function(X[y>1.5]).ravel()
        d1 = clf.decision_function(X[y==processLabels[sigLegend]])
        d2 = clf.decision_function(X[y==processLabels[bg0Legend]])
        d3 = clf.decision_function(X[y==processLabels[bg1Legend]])
        #d1 = clf.decision_function(X[y==0])
        #d2 = clf.decision_function(X[y==1])
        #d3 = clf.decision_function(X[y==2])
        decisions += [d1[:, iColForSig], d2[:, iColForSig], d3[:, iColForSig]]
        #decisions += [d1[:, 0], d2[:, 0], d3[:, 0]]

    #low = min(np.min(d) for d in decisions)
    #high = max(np.max(d) for d in decisions)
    low = -15 
    high = 15 
    low_high = (low, high)
    fig=plt.figure(figsize=(8,5.5))
    fig.patch.set_color('white')
    plt.hist(decisions[0], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density = True, label='{} (train)'.format(sigLegend))
    plt.hist(decisions[1], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density = True, label='{} (train)'.format(bg0Legend))
    plt.hist(decisions[2], color='g', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density = True, label='{} (train)'.format(bg1Legend))
 
    sigScore_pass = decisions[0]>=0
    sigScore_failed = decisions[0]<0
    print("Signal with score>=0", sigScore_pass.sum()) 
    print("Signal with score<0", sigScore_failed.sum()) 

    hist, bins = np.histogram(decisions[3], bins=bins, range=low_high, density=True)
    scale = len(decisions[3])/sum(hist)
    err = np.sqrt(hist*scale)/scale

    width = (bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='{} (test)'.format(sigLegend))

    hist, bins = np.histogram(decisions[4], bins=bins, range=low_high, density=True)
    scale = len(decisions[4])/sum(hist)
    err = np.sqrt(hist*scale)/scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='{} (test)'.format(bg0Legend))

    hist, bins = np.histogram(decisions[5], bins=bins, range=low_high, density=True)
    scale = len(decisions[5])/sum(hist)
    err = np.sqrt(hist*scale)/scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='g', label='{} (test)'.format(bg1Legend))
  
  
    plt.xlabel("BDT score")
    plt.ylabel("Normalized Unit")
    plt.legend(loc='best')
    plt.savefig(output+"/"+savepath+"/BDTscore"+label+".pdf")
#    plt.show()


def run_grid_search(outdir, bdt, x, y):
    logging.info('starting hyper-parameter optimization')
    param_grid = {"n_estimators": [50,100,800,1000], 'learning_rate': [0.01,0.1,0.5]}

    clf = grid_search.GridSearchCV(bdt, param_grid, cv=CV, scoring='roc_auc', n_jobs=NJOBS, verbosity=2)
    clf.fit(x, y)

    out = '\nHyper-parameter optimization\n'
    out += '============================\n\n'
    out += 'Best estimator: {}\n'.format(clf.best_estimator_)
    out += '\nFull Scores\n'
    out += '-----------\n\n'
    for params, mean_score, scores in clf.grid_scores_:
	    out += u'{:0.4f} (±{:0.4f}) for {}\n'.format(mean_score, scores.std(), params)
    with codecs.open(os.path.join(outdir, "log-hyper-parameters.txt"), "w", encoding="utf8") as fd:
	    fd.write(out)

def plot_inputs(outdir, vars, branch_labels, sig, sig_w, bkg, bkg_w, bkg2, bkg2_w, sigLegend, bg0Legend, bg1Legend):
    for n, var in enumerate(vars):
        _, bins = np.histogram(np.concatenate((sig[:, n], bkg[:, n],bkg2[:, n])), bins=40)
        sns.distplot(sig[:, n], hist_kws={'weights': sig_w}, bins=bins, kde=False, norm_hist=True, color='orange', label='{}'.format(sigLegend))
        sns.distplot(bkg[:, n], hist_kws={'weights': bkg_w}, bins=bins, kde=False, norm_hist=True, color='b', label='{}'.format(bg0Legend))
        sns.distplot(bkg2[:, n], hist_kws={'weights': bkg2_w}, bins=bins, kde=False, norm_hist=True, color='g', label='{}'.format(bg1Legend))
        #plt.title(var)
        plt.legend()
        if var=="first_dEs":
           plt.subplots_adjust(left=0.15) 
        plt.xlabel('{}'.format(branch_labels[var]),  loc='right')
        plt.ylabel('Entries')
        plt.savefig(os.path.join(outdir, 'input_{}.pdf'.format(var)))
        plt.savefig(os.path.join(outdir, 'input_{}.pdf'.format(var)))
        plt.close()

def plot_correlations(outdir, vars, branch_labels, sig, bkg0, bkg1):
    for data, label in ((sig, "Signal"), (bkg0, "Background0"), (bkg1, "Background1")):
        labels = [] 
        for key in branch_labels:
            labels.append(branch_labels[key]) 
        d = pd.DataFrame(data, columns=labels)
        sns.heatmap(d.corr(), annot=True, fmt=".2f", linewidth=.5)
        plt.title(label + " Correlations")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'correlations_{}.pdf'.format(label.lower())))
        plt.savefig(os.path.join(outdir, 'correlations_{}.pdf'.format(label.lower())))
        plt.close()


def plot_learning_curve(outdir, bdt, x, y):
    logging.info("creating learning curve")
    train_sizes, train_scores, test_scores = learning_curve(bdt,
								x,
								y,
		                                                cv=ShuffleSplit(len(x),
		                                                n_iter=100,
		                                                test_size=1.0 / CV),
		                                            	n_jobs=NJOBS,
								verbosity=2,
		                                            	train_sizes=np.linspace(.1, 1., 7),
		                                            	scoring='roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes,
		     train_scores_mean - train_scores_std,
		     train_scores_mean + train_scores_std,
		     alpha=.2, color='r')
    plt.fill_between(train_sizes,
		     test_scores_mean - test_scores_std,
		     test_scores_mean + test_scores_std,
		     alpha=.2, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

    plt.xlabel("Sample size")
    plt.ylabel("Score (ROC area)")

    plt.legend()
    plt.savefig(os.path.join(outdir, 'learning-curve.pdf'))
    plt.savefig(os.path.join(outdir, 'learning-curve.pdf'))
    plt.close()

if __name__ == '__main__':
    print('start')
    main()
