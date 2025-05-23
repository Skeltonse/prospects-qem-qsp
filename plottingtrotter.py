import pennylane as qml
from QSP_sims_clean import exponential_model, linear_model, richardson, MITIGATION
import os
import numpy as np
from scipy.optimize import curve_fit
import pickle
from matplotlib import pyplot as plt

import functions.laur_poly_fcns as lpf
import tikzplotlib

import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
plt.rcParams.update({'text.usetex': True,'font.family': 'serif',})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{braket}')

'''DEFINE THE FIGURE AND DOMAIN'''
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams['font.size'] = 12
colorlist=['tab:orange', 'tab:red', 'tab:green','tab:cyan' , 'tab:purple', ]
stylelist=['.', '_', '.', '.', '.', '.']
linelist=['solid', '-', 'solid', '-', 'solid']


def PLOTTING_trotter(times,pU, scaling,  Hlabel="SK4", lchoice="degree", degree=13, eorder=3):

    # do mitigation and generate mitigated and noisy values for the plots
    mitigated = []
    noisy = []
    variances = []
    ideal = []
    bias = []
    MSE = []
    depth=[]
    for t in times:
        tau = t
        pathname="plotting.py"
        if lchoice=='degree':
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)) +"_deg_" +str(degree) + "_t_" + str(tau) 
        elif lchoice=='order':
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)) +"_eorder_" + str(eorder) + "_t_" + str(tau) 
        elif prec>=5:
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" + "1.0e-"+str(prec) + "_t_" + str(tau)
        else:
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" +str(10**(-prec)) + "_t_" + str(tau)
        current_path = os.path.abspath(__file__)
        parentdir=os.path.dirname(current_path)
        datadir=os.path.join(parentdir, "benchmark_data")
        
        with open(os.path.join(datadir, filename+'.pkl'), 'rb') as f:
            datadict = pickle.load(f)
        # return datadict['Hamiltonian']
        
        mit, var = MITIGATION(scaling, tau, datadict)
        

        # print(datadict['noisyexpdict_'+str(0)].keys())
        nois = datadict['noisyexpdict_'+str(tau)]["noisyexpectationarray_"+str(1)]
        idi = datadict['noisyexpdict_'+str(tau)]["idealexpectationvalue"]
        b = mit-idi
        m = var+b**2
        mitigated.append(mit)
        variances.append(var)
        noisy.append(nois)
        ideal.append(idi)
        bias.append(b)
        MSE.append(m)
        depth.append(datadict['noisyexpdict_'+str(tau)]['depth'])

    dict={'ideal': ideal, 'noisy':noisy, 'mitigated': mitigated, 'variances': variances, 'bias': bias, 'MSE': MSE, 'depth': depth}
    return dict


def plotting_choice_trotter(times, prec, scaling, l, pU, Hlabel="SK4", exttrue=True, lchoice='degree', degree=13):

    # do mitigation and generate mitigated and noisy values for the plots
    values = [1, 1.1, 1.25, 1.3, 1.5, 1.75,  2, 3]
    indices = [values.index(value) for value in scaling]
    mitigated_linear = []
    mitigated_exp = []
    mitigated_richardson = []
    ideal=[]
    noisy=[]
    variances_lin = []
    variances_exp = []
    variances_rid=[]
    bias_lin = []
    bias_exp = []
    bias_rid=[]
    MSE_lin = []
    MSE_exp = []
    MSE_rid=[]
    full_error = []
    bias = []
    for t in times:
        tau = t
       
        pathname="plotting.py"
        if lchoice=='degree':
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)) +"_deg_" +str(degree) + "_t_" + str(tau) 
        elif lchoice=='order':
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)) +"_eorder_" + str(5) + "_t_" + str(tau) 
        elif prec>=5:
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" + "1.0e-"+str(prec) + "_t_" + str(tau)
        else:
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" +str(10**(-prec)) + "_t_" + str(tau)
        
        current_path = os.path.abspath(__file__)
        parentdir=os.path.dirname(current_path)
        datadir=os.path.join(parentdir, "benchmark_data")
        with open(os.path.join(datadir, filename+'.pkl'), 'rb') as f:
            datadict = pickle.load(f)
        nois = datadict['noisyexpdict_'+str(tau)]["noisyexpectationarray_"+str(1)]
        
        idi=datadict['noisyexpdict_'+str(tau)]["idealexpectationvalue"]
        
        noisy.append(nois)
        ideal.append(idi)
        variance = datadict['noisyexpdict_'+str(tau)]['variance']
        variance = [variance[i]/5e+6 for i in indices]
        y = []
        for c in scaling:
            expectation_noisy = datadict['noisyexpdict_'+str(tau)]["noisyexpectationarray_"+str(c)]
            y.append(expectation_noisy)

        # linear fit
        params, _ = curve_fit(linear_model, scaling, y)
        new_x = 0 
        mit_linear = linear_model(new_x, *params)
        mitigated_linear.append(mit_linear)
        var_lin = sum(variance)*params[1]**2
        variances_lin.append(var_lin)
        bias_lin.append(mit_linear-idi)
        MSE_lin.append(var_lin+(mit_linear-idi)**2)
        

        # richardosn fit
        mit_richardson, parameters = richardson(y, scaling)
        mitigated_richardson.append(mit_richardson)
        var_rid = variance[0]*params[0]**2+variance[1]*params[1]**2
        variances_rid.append(var_rid)
        
        
        bias_rid.append(mit_richardson-idi)
        MSE_rid.append(var_rid+(mit_richardson-idi)**2)
    
        # exponential fit
        if exttrue==True or exttrue=='all':
            params, _ = curve_fit(exponential_model, scaling, y, maxfev=1000)
            new_x = 0 
            mit_exp = exponential_model(new_x, *params)
            mitigated_exp.append(mit_exp)
            var_exp = params[1]*np.exp(-1*params[1]*mit_exp)**2*sum(variance)
            variances_exp.append(var_exp)
            bias_exp.append(mit_exp-idi)
            MSE_exp.append(var_exp+(mit_exp-idi)**2)
    

    dict={'ideal': ideal, 'noisy':noisy, 'mitigated_linear': mitigated_linear, 'variances_lin': variances_lin, 'bias_lin': bias_lin, 'MSE_lin': MSE_lin, 'mitigated_richardson': mitigated_richardson, 'variances_rid': variances_rid, 'bias_rid': bias_rid, 'MSE_rid': MSE_rid}
    if exttrue==True:
        dict['mitigated_exp']=mitigated_exp
        dict['MSE_exp']=MSE_exp
        return dict
    else:
        return dict


taulisttrotter=np.append(np.arange(0.1, 5, 0.1), np.arange(5.2, 20, 0.2))
indicestrotter=np.argsort(taulisttrotter)
taulisttrotter=taulisttrotter[indicestrotter]


def APPENDIX_COMP_PLOT(taulist, hlabel="_shots_TI4_dt01", ifsave=True):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), )
    dict= plotting_choice_trotter(taulist, 4, [1, 1.25, 1.5], 0, 0.0001, Hlabel=hlabel, exttrue=True, lchoice='degree', degree=5)
    axs[0, 0].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[0, 0].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[0, 0].scatter(taulist, dict['mitigated_linear'], color=colorlist[2],marker="2")
    axs[0, 0].scatter(taulist, dict['mitigated_richardson'], color=colorlist[3],  marker=".")
    axs[0, 0].scatter(taulist, dict['mitigated_exp'],color=colorlist[4], marker="3")
    axs[0, 0].set_title(r'$p=0.0001$')

    dict= plotting_choice_trotter(taulist, 4, [1, 1.25, 1.5], 0, 0.001, Hlabel=hlabel, exttrue=True, lchoice='degree', degree=5)
    axs[0, 1].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[0, 1].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[0, 1].scatter(taulist, dict['mitigated_linear'], color=colorlist[2],marker="2")
    axs[0, 1].scatter(taulist, dict['mitigated_richardson'], color=colorlist[3],  marker=".")
    axs[0, 1].scatter(taulist, dict['mitigated_exp'],color=colorlist[4], marker="3")
    axs[0, 1].set_title(r'$p=0.001$')

    dict= plotting_choice_trotter(taulist, 4, [1, 1.25, 1.5], 0, 0.01, Hlabel=hlabel, exttrue=False, lchoice='degree', degree=5)
    axs[0, 2].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[0, 2].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[0, 2].scatter(taulist, dict['mitigated_linear'], color=colorlist[2],marker="2")
    
    axs[0, 2].set_title(r'$p=0.01$')

    dict= plotting_choice_trotter(taulist, 4, [1, 2, 3], 0, 0.0001, Hlabel=hlabel, exttrue=True, lchoice='degree', degree=5)
    axs[1, 0].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[1, 0].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[1, 0].scatter(taulist, dict['mitigated_linear'], color=colorlist[2],marker="2")
    axs[1, 0].scatter(taulist, dict['mitigated_exp'],color=colorlist[4], marker="3")
    axs[1, 0].set_title(r'$p=0.0001$') 
    
    dict= plotting_choice_trotter(taulist, 4, [1, 2, 3], 0, 0.001, Hlabel=hlabel, exttrue=True, lchoice='degree', degree=5)
    axs[1, 1].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[1, 1].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[1, 1].scatter(taulist, dict['mitigated_linear'], color=colorlist[2],marker="2")
    axs[1, 1].scatter(taulist, dict['mitigated_richardson'], color=colorlist[3],  marker=".")
    axs[1, 1].scatter(taulist, dict['mitigated_exp'],color=colorlist[4], marker="3")
    axs[1, 1].set_title(r'$p=0.001$') 

    dict= plotting_choice_trotter(taulist, 4, [1, 2, 3], 0, 0.01, Hlabel=hlabel, exttrue=False, lchoice='degree', degree=5)
    axs[1, 2].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0],label={'ideal'} )
    axs[1, 2].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" , label={'noisy'})
    axs[1, 2].scatter(taulist, dict['mitigated_linear'], color=colorlist[2],marker="2", label={'linear'})
    axs[1, 2].scatter(taulist, dict['mitigated_richardson'], color=colorlist[3],  marker=".", label={'Richardson'})
    axs[1, 2].scatter(taulist, dict['mitigated_exp'],color=colorlist[4], marker="3", label={'exp'})
    axs[1, 2].set_title(r'$p=0.01$') 


    fig.supxlabel(r'simulation time $\tau$')
    fig.supylabel(r'$\langle I_0\otimes Z_1\otimes Z_2\otimes I_3\rangle$')

    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")
    
    plt.show()
    return 

def MSE_SCALING_COMP(taulist, hlabel="_shots_TI4_shots", ifsave=True, totau=20, d=31):
    fig, axs = plt.subplots(3, 3, figsize=(24, 20), )
    
    dict25= plotting_choice_trotter(taulist, 4, [1, 1.25, 1.5], 0, 0.0001, Hlabel=hlabel, exttrue=True, lchoice='degree', degree=5)
    dict1= plotting_choice_trotter(taulist, 4, [1, 2, 3], 0, 0.0001, Hlabel=hlabel, exttrue=True,  lchoice='degree', degree=5)
    axs[0, 0].plot(taulist, dict25['MSE_lin'], color=colorlist[0],marker="2", label="[1, 1.25, 1.5]")
    axs[0, 0].plot(taulist, dict1['MSE_lin'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    axs[0, 0].set_title(r'$p=0.0001, lin fit$')
    axs[1, 0].plot(taulist, dict25['MSE_rid'], color=colorlist[0],marker="2", label="[1, 1.25, 1.5]")
    axs[1, 0].plot(taulist, dict1['MSE_rid'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    axs[1, 0].set_title(r'$p=0.0001, rid fit$')
    axs[0, 0].legend()
    axs[2,0].plot(taulist, dict25['MSE_exp'],color=colorlist[0], marker="2")
    axs[2, 0].plot(taulist, dict1['MSE_exp'],color=colorlist[1], marker="3")
    axs[2, 0].set_title(r'$p=0.0001, exp fit$')

    dict25= plotting_choice_trotter(taulist, 4, [1, 1.25, 1.5], 0, 0.001, Hlabel=hlabel, exttrue=True,  lchoice='degree', degree=5)
    dict1= plotting_choice_trotter(taulist, 4, [1, 2, 3], 0, 0.001, Hlabel=hlabel, exttrue=True,  lchoice='degree', degree=5)
    axs[0, 1].plot(taulist, dict25['MSE_lin'], color=colorlist[0],marker="2", label="[1, 1.25, 1.5]")
    axs[0, 1].plot(taulist, dict1['MSE_lin'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    axs[0, 1].set_title(r'$p=0.001, lin fit$')
    axs[1, 1].plot(taulist, dict25['MSE_rid'], color=colorlist[0],marker="2", label="[1, 1.25, 1.5]")
    axs[1, 1].plot(taulist, dict1['MSE_rid'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    axs[1, 1].set_title(r'$p=0.001, rid fit$')
    axs[2,1].plot(taulist, dict25['MSE_exp'],color=colorlist[0], marker="2")
    axs[2, 1].plot(taulist, dict1['MSE_exp'],color=colorlist[1], marker="3")
    axs[2, 1].set_title(r'$p=0.001, exp fit$')

    dict25= plotting_choice_trotter(taulist, 4, [1, 1.25, 1.5], 0, 0.01, Hlabel=hlabel, exttrue=False,  lchoice='degree', degree=5)
    dict1= plotting_choice_trotter(taulist, 4, [1, 2, 3], 0, 0.01, Hlabel=hlabel, exttrue=False,  lchoice='degree', degree=5)
    axs[0, 2].plot(taulist, dict25['MSE_lin'], color=colorlist[0],marker="2", label="[1, 1.25, 1.5]")
    axs[0, 2].plot(taulist, dict1['MSE_lin'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    axs[0, 2].set_title(r'$p=0.01, lin fit$')
    axs[1, 2].plot(taulist, dict25['MSE_rid'], color=colorlist[0],marker="2", label="[1, 1.25, 1.5]")
    axs[1, 2].plot(taulist, dict1['MSE_rid'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    axs[1, 2].set_title(r'$p=0.01, rid fit$')
 
    plt.show()
    return 

def IDEALMITCOMPPLOTALLQ(taulist, ifsave=False):
    fig, axs = plt.subplots(3, 3, figsize=(18, 12), )
    
    ###4 qubits
    dict= plotting_choice_trotter(taulist, 4, [1, 2, 3], 0, 0.0001, Hlabel="TI4_dt01_fixed", exttrue=True, lchoice='degree',degree=5)
    axs[0, 0].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[0, 0].scatter(taulist, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[0, 0].set_title(r'$p=0.0001$')

    dict= plotting_choice_trotter(taulist, 4,  [1, 2, 3], 0, 0.001, Hlabel="TI4_dt01_fixed", exttrue=True, lchoice='degree',degree=5 )
    axs[0, 1].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[0, 1].scatter(taulist, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[0, 1].set_title(r'$p=0.001$')

    dict= plotting_choice_trotter(taulist, 4, [1, 1.25, 1.5], 0, 0.01, Hlabel="TI4_dt01_fixed", exttrue=False, lchoice='degree',degree=5 )
    axs[0, 2].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[0, 2].scatter(taulist, dict['mitigated_richardson'], color=colorlist[2],marker="2")
    axs[0, 2].set_title(r'$p=0.01$')

    ###6 qubits
    dict= plotting_choice_trotter(taulist, 4, [1, 2, 3], 0, 0.0001, Hlabel="TI6_dt01_fixed", exttrue=True, lchoice='degree',degree=5 )
    axs[1, 0].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[1, 0].scatter(taulist, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[1, 0].set_title(r'$p=0.0001$') 
    
    dict= plotting_choice_trotter(taulist, 4,  [1, 2, 3], 0, 0.001, Hlabel="TI6_dt01_fixed", exttrue=True, lchoice='degree',degree=5 )
    axs[1, 1].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[1, 1].scatter(taulist, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[1, 1].set_title(r'$p=0.001$') 

    dict= plotting_choice_trotter(taulist, 4, [1, 1.25, 1.5], 0, 0.01, Hlabel="TI6_dt01_fixed", exttrue=False,lchoice='degree',degree=5 )
    axs[1, 2].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0],label={'ideal'} )
    axs[1, 2].scatter(taulist, dict['mitigated_richardson'], color=colorlist[2],marker="2", label={'linear'})
    axs[1, 2].set_title(r'$p=0.01$') 

    ###8 qubits
    taulist8q=taulist#np.arange(0.1, 5, 0.1)
    dict= plotting_choice_trotter(taulist8q, 4, [1, 2, 3], 0, 0.0001, Hlabel="TI8_dt01_fixed", exttrue=True, lchoice='degree',degree=5 )
    axs[2, 0].plot(taulist8q, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[2, 0].scatter(taulist8q, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[2, 0].set_title(r'$p=0.0001$') 
    
    dict= plotting_choice_trotter(taulist8q, 4,  [1, 2, 3], 0, 0.001, Hlabel="TI8_dt01_fixed", exttrue=True, lchoice='degree',degree=5 )
    axs[2, 1].plot(taulist8q, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[2, 1].scatter(taulist8q, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[2, 1].set_title(r'$p=0.001$') 

    dict= plotting_choice_trotter(taulist8q, 4,  [1, 1.25, 1.5], 0, 0.01, Hlabel="TI8_dt01_fixed", exttrue=False, lchoice='degree',degree=5 )
    axs[2, 2].plot(taulist8q, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[2, 2].scatter(taulist8q, dict['mitigated_richardson'], color=colorlist[2],marker="2")
    axs[2, 2].set_title(r'$p=0.01$') 

    fig.supxlabel(r'simulation time $\tau$')
    fig.supylabel(r'$\langle I_0\otimes Z_1\otimes Z_2\otimes I_3\rangle$')
    
    savetitle="idealmitcompplotallqtrotter.tex"
    pathname="plottingtrotter.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")
    plt.show()
    return 
IDEALMITCOMPPLOTALLQ(taulisttrotter, ifsave=False)

def plottingnoises(tau, prec, scaling, l, noises, Hlabel="SK4", exttrue=True, lchoice="degree", degree=13):

    # do mitigation and generate mitigated and noisy values for the plots
    values = [1, 1.1, 1.25, 1.3, 1.5, 1.75,  2, 3]
    indices = [values.index(value) for value in scaling]
    mitigated_linear = []
    mitigated_exp = []
    mitigated_richardson = []
    ideal=[]
    noisy=[]
    variances_lin = []
    variances_exp = []
    variances_rid=[]
    bias_lin = []
    bias_exp = []
    bias_rid=[]
    MSE_lin = []
    MSE_exp = []
    MSE_rid=[]
    full_error = []
    bias = []
    for pU in noises:
        
        pathname="plotting.py"
        if lchoice=='degree':
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)) +"_deg_" +str(degree) + "_t_" + str(tau) 
        elif lchoice=='order':
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)) +"_eorder_" + str(5) + "_t_" + str(tau) 
        if prec>=5:
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" + "1.0e-"+str(prec) + "_t_" + str(tau)
        else:
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" +str(10**(-prec)) + "_t_" + str(tau)
        current_path = os.path.abspath(__file__)
        parentdir=os.path.dirname(current_path)
        datadir=os.path.join(parentdir, "benchmark_data")
        with open(os.path.join(datadir, filename+'.pkl'), 'rb') as f:
            datadict = pickle.load(f)
        nois = datadict['noisyexpdict_'+str(tau)]["noisyexpectationarray_"+str(1)]
        idi=datadict['noisyexpdict_'+str(tau)]["idealexpectationvalue"]
        noisy.append(nois)
        ideal.append(idi)
        variance = datadict['noisyexpdict_'+str(tau)]['variance']
        variance = variance
        variance = [variance[i]/5e6 for i in indices]
        y = []
        for c in scaling:
            expectation_noisy = datadict['noisyexpdict_'+str(l)]["noisyexpectationarray_"+str(c)]
            y.append(expectation_noisy)

        # linear fit
        params, _ = curve_fit(linear_model, scaling, y)
        new_x = 0 
        mit_linear = linear_model(new_x, *params)
        mitigated_linear.append(mit_linear)
        var_lin = sum(variance)*params[1]**2
        #var_lin = sum(variance)
        variances_lin.append(var_lin)
        bias_lin.append(mit_linear-idi)
        MSE_lin.append(var_lin-(mit_linear-idi)**2)
        

        # richardosn fit
        mit_richardson, parameters = richardson(y, scaling)
        mitigated_richardson.append(mit_richardson)
        var_rid = variance[0]*params[0]**2+variance[1]*params[1]**2
        #var_rid = sum(variance)
        variances_rid.append(var_rid)
        
        
        bias_rid.append(mit_richardson-idi)
        MSE_rid.append(var_rid-(mit_richardson-idi)**2)
    
        # exponential fit
        if exttrue==True:
            params, _ = curve_fit(exponential_model, scaling, y, maxfev=1000)
            new_x = 0 
            mit_exp = exponential_model(new_x, *params)
            mitigated_exp.append(mit_exp)
            var_exp = params[1]*np.exp(-1*params[1]*mit_exp)**2*sum(variance)
            #var_exp = sum(variance)
            variances_exp.append(var_exp)
            bias_exp.append(mit_exp-idi)
            MSE_exp.append(var_exp-(mit_exp-idi)**2)
    


    if exttrue==True:
        return ideal, noisy, mitigated_linear, mitigated_richardson, mitigated_exp#, variances_lin, bias_lin, MSE_lin, mitigated_exp, variances_exp, bias_exp, MSE_exp, mitigated_richardson, variances_rid, bias_rid, MSE_rid
    elif exttrue=='vars':
        return  ideal, noisy, variances_lin, variances_rid
    else:
        return ideal, noisy, mitigated_linear, mitigated_richardson

def EXPvsNOISEPLOT():
    noisearray=np.linspace(0.001, 0.3, 30)
    

    fig, axs = plt.subplots(1, figsize=(18, 12), )
    fig.supxlabel(r'noise parameter $p$')
    fig.supylabel(r'$\langle I_0\otimes Z_1\otimes Z_2\otimes I_3\rangle$')
    ylabels=[r'scaling $\left[1, 1.25, 1.5\right]$', r'scaling $\left[1, 2, 3\right]$']
    
    taulist=[1, 5, 15]
    for j in range(3):
        ideal, noisy, mitigated_linear, mitigated_richardson=plottingnoises(taulist[j], 4, [1, 1.25, 1.5], 1, noisearray, Hlabel="TI4", exttrue=False)
        axs.plot(noisearray,mitigated_richardson, color=colorlist[j], linestyle=linelist[j], label=str(taulist[j]))
        

    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    plt.legend()
    plt.show()


def GETDICT(times, l, pU, Hlabel="SK4", lchoice="degree", degree=5):
    shots=[]
    depth=[]
    for t in times:
        tau = t
        if lchoice=='degree':
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(1000*pU)) +"_deg_" +str(degree) + "_t_" + str(tau) 
        elif lchoice=='order':
            filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(1000*pU)) +"_eorder_" + str(5) + "_t_" + str(tau) 
       
        current_path = os.path.abspath(__file__)
        parentdir=os.path.dirname(current_path)
        datadir=os.path.join(parentdir, "benchmark_data")
        
        with open(os.path.join(datadir, filename+'.pkl'), 'rb') as f:
            datadict = pickle.load(f)
        shots.append(datadict['noisyexpdict_'+str(tau)]['shots'])
        depth.append(datadict['noisyexpdict_'+str(tau)]['depth'])
    return shots, depth

def MvsDPLOTS(taulist, Hlabel="TI4_dt01"):
    
    fig, ax = plt.subplots(2 , 1, figsize=(10, 10))

    shots, depth=GETDICT(taulist,0, 0.001, Hlabel="TI4_dt01")
    ax[0].plot(depth, np.log10(shots))
    ax[0].plot(depth, (6)*np.ones(len(shots)))


    const = -2 * np.log10(0.001)
    M_eisert = [x * const for x in depth]

    ax[1].plot(depth, M_eisert)
    ax[1].plot(depth, (6)*np.ones(len(shots)))

    
    fig.supylabel(r'$\log_{10}(M)$')
    fig.supxlabel(r'depth $D$')

    pathname="plottingtrotter.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    tikzplotlib.save(os.path.join(save_path, "shotsvstau.tex"), flavor="context")
    plt.show()
    return


def DEPTHvsTAU(taulist, prec=4, pU=0.001, dor=1, Hlabel="TI4", lchoice="degree", degree=13):

    # do mitigation and generate mitigated and noisy values for the plots
    degreelist=[]

    for tau in taulist:
        pathname="plotting.py"
        if lchoice=='degree':
            filename="trotternoisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_deg_" +str(degree) + "_t_" + str(tau) 
        elif lchoice=='order':
            filename="trotternoisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_eorder_" + str(5) + "_t_" + str(tau) 
        elif prec>=5:
            filename="trotternoisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" + "1.0e-"+str(prec) + "_t_" + str(tau)
        else:
            filename="trotternoisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" +str(10**(-prec)) + "_t_" + str(tau)
        current_path = os.path.abspath(__file__)
        parentdir=os.path.dirname(current_path)
        datadir=os.path.join(parentdir, "benchmark_data")
        with open(os.path.join(datadir, filename+'.pkl'), 'rb') as f:
            datadict = pickle.load(f)
        
        degreelist.append(datadict['QSPdict']['n'])

    depthlist=[2*n+1+2*n*dor for n in degreelist]
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), )
    depthlist1=[2*n+1+2*n*1 for n in degreelist]
    dor=10
    depthlist10=[2*n+1+2*n*10 for n in degreelist]
    dor=60
    depthlist60=[2*n+1+2*n*60 for n in degreelist]
    axs[1].plot(taulist, depthlist1, label=r'depth, $d_o=1$')
    axs[1].plot(taulist, depthlist10, label=r'depth, $d_o=10$')
    axs[1].plot(taulist, depthlist60, label=r'depth, $d_o=60$')
    # plt.legend()
    axs[0].plot(taulist, degreelist, label='degree')
    # plt.legend()
    fig.supxlabel(r'simulation time $\tau$')
    axs[0].set_ylabel(r'polynomial degree $n$')
    axs[1].set_ylabel(r'circuit depth $d$')

    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    tikzplotlib.save(os.path.join(save_path, "depthvstau2.tex"), flavor="context")
    
    plt.legend()
    plt.show()
    return 
