from QSP_sims_clean import MITIGATION, exponential_model, linear_model, richardson
import os
import numpy as np
from scipy.optimize import curve_fit
import pickle
from matplotlib import pyplot as plt

import functions.laur_poly_fcns as lpf
from plottingtrotter import PLOTTING_trotter, plotting_choice_trotter
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

taulistp25 = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0]
taulist5p25 = [5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75,  15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0]
taulistp1 = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 2.3, 1.4, 1.6, 1.7, 1.8, 1.9, 2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9, 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8, 3.9, 4.1, 4.2, 4.3, 4.4,4.6, 4.7, 4.8, 4.9]
taulist8p25=[5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75]

taulistqsp=np.array(taulistp1+taulistp25)
taulistprecise=np.array(taulistp1+taulist5p25)
taulisttrotter=np.append(np.arange(0.1, 5, 0.1), np.arange(5.2, 20, 0.2))
indicesqsp=np.argsort(taulistqsp)
taulistqsp=taulistqsp[indicesqsp]
indicestrotter=np.argsort(taulisttrotter)
taulisttrotter=taulisttrotter[indicestrotter]
taulistlongqsp = [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0, 325.0, 350.0, 375.0, 390.0]


def PLOTTINGQSP(times, prec, scaling, l, pU, Hlabel="SK4", lchoice="degree", degree=13, eorder=3):

    # do mitigation and generate mitigated and noisy values for the plots
    mitigated = []
    noisy = []
    variances = []
    ideal = []
    bias = []
    depth=[]
    MSE = []
    for t in times:
        tau = t
        pathname="plotting.py"
        if lchoice=='degree':
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_deg_" +str(degree) + "_t_" + str(tau) 
        elif lchoice=='order':
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_eorder_" + str(eorder) + "_t_" + str(tau) 
        elif prec>=5:
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" + "1.0e-"+str(prec) + "_t_" + str(tau)
        else:
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" +str(10**(-prec)) + "_t_" + str(tau)
        current_path = os.path.abspath(__file__)
        parentdir=os.path.dirname(current_path)
        datadir=os.path.join(parentdir, "benchmark_data")
        with open(os.path.join(datadir, filename+'.pkl'), 'rb') as f:
            datadict = pickle.load(f)
        # print(datadict['Hdict'].keys())
        # return datadict['Hdict']['H']
    
        mit, var = MITIGATION(scaling, 0, datadict)
        nois = datadict['noisyexpdict_'+str(0)]["noisyexpectationarray_"+str(1)]
        id = datadict['classicalvals'][0]
        b = mit-id
        m = var+b**2
        mitigated.append(mit)
        variances.append(var)
        noisy.append(nois)
        ideal.append(id)
        bias.append(b)
        MSE.append(m)
        polydegree=datadict['QSPdict']['n']
        depth.append(2*polydegree+1)

    dict={'ideal': ideal, 'noisy':noisy, 'mitigated': mitigated, 'variances': variances, 'bias': bias, 'MSE': MSE, 'depth': depth}
    return dict

def GETVARS(times, prec, scaling, l, pU, Hlabel="SK4", lchoice="degree", degree=13, eorder=3):

    # do mitigation and generate mitigated and noisy values for the plots
    mitigated = []
    noisy = []
    variances = []
    ideal = []
    bias = []
    MSE = []
    for t in times:
        tau = t
        pathname="plotting.py"
        if lchoice=='degree':
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_deg_" +str(degree) + "_t_" + str(tau) 
        elif lchoice=='order':
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_eorder_" + str(eorder) + "_t_" + str(tau) 
        elif prec>=5:
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" + "1.0e-"+str(prec) + "_t_" + str(tau)
        else:
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" +str(10**(-prec)) + "_t_" + str(tau)
        current_path = os.path.abspath(__file__)
        parentdir=os.path.dirname(current_path)
        datadir=os.path.join(parentdir, "benchmark_data")
        with open(os.path.join(datadir, filename+'.pkl'), 'rb') as f:
            datadict = pickle.load(f)
        # print(datadict['Hdict'].keys())
        # return datadict['Hdict']['H']
        
        #mit, var = MITIGATION(scaling, 0, datadict)
    #     nois = datadict['noisyexpdict_'+str(0)]["noisyexpectationarray_"+str(1)]
    #     idi = datadict['classicalvals'][0]
    #     b = mit-idi
    #     m = var+b**2
    #     mitigated.append(mit)
    #     variances.append(var)
    #     noisy.append(nois)
    #     ideal.append(id)
    #     bias.append(b)
    #     MSE.append(m)

    # dict={'ideal': ideal, 'noisy':noisy, 'mitigated': mitigated, 'variances': variances, 'bias': bias, 'MSE': MSE}
    # return datadict['noisyexpdict_'+str(0)]['variance'][0]
    return datadict['noisyexpdict_'+str(0)]["noisyexpectationarray_1"]

def plotting_choice_QSP(times, prec, scaling, l, pU, Hlabel="SK4", exttrue=True, lchoice='degree', degree=13, eorder=3):
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
        print(t)
        tau = t
       
        pathname="plotting.py"
        if lchoice=='degree':
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_deg_" +str(degree) + "_t_" + str(tau) 
        elif lchoice=='order':
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_eorder_" + str(eorder) + "_t_" + str(tau) 
        elif prec>=5:
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" + "1.0e-"+str(prec) + "_t_" + str(tau)
        else:
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" +str(10**(-prec)) + "_t_" + str(tau)
        
        current_path = os.path.abspath(__file__)
        parentdir=os.path.dirname(current_path)
        datadir=os.path.join(parentdir, "benchmark_data")
        with open(os.path.join(datadir, filename+'.pkl'), 'rb') as f:
            datadict = pickle.load(f)
        nois = datadict['noisyexpdict_'+str(l)]["noisyexpectationarray_"+str(1)]
        
        ###due to general dumbassery in data saving, recalculate the classical values###
        idi=datadict['classicalvals'][l]
        
        noisy.append(nois)
        ideal.append(idi)
        variance = datadict['noisyexpdict_'+str(l)]['variance']
        # variance = variance
        # print(variance)
        variance = [variance[i]/5e+6 for i in indices]
        # variance = [variance[i] for i in indices]
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
        MSE_lin.append(var_lin+(mit_linear-idi)**2)
        

        # richardosn fit
        mit_richardson, parameters = richardson(y, scaling)
        mitigated_richardson.append(mit_richardson)
        var_rid = variance[0]*params[0]**2+variance[1]*params[1]**2
        #var_rid = sum(variance)
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
            #var_exp = sum(variance)
            variances_exp.append(var_exp)
            bias_exp.append(mit_exp-idi)
            MSE_exp.append(var_exp+(mit_exp-idi)**2)
    
    dict={'ideal': ideal, 'noisy':noisy, 'mitigated_linear': mitigated_linear, 'variances_lin': variances_lin, 'bias_lin': bias_lin, 'MSE_lin': MSE_lin, \
        'mitigated_richardson': mitigated_richardson, 'variances_rid': variances_rid, 'bias_rid': bias_rid, 'MSE_rid': MSE_rid, 'var_lin': var_lin, 'var_rid': var_rid, 'var_exp': var_exp}
    if exttrue==True:
        dict['mitigated_exp']=mitigated_exp
        dict['MSE_exp']=MSE_exp
        dict['variances_exp']=variances_exp
        return dict
    else:
        return dict

# ####BIG APPENDIX ARRAY####
def MAINTEXTPLOT(taulistqsp, p=0.0001, scalingqsp=[1, 1.25, 1.5], scalingtrotter=[1, 1.25, 1.5]):
    # fig, axs = plt.subplots(1, figsize=(18, 12), )
    # # fig, ax = plt.subplots(1 , 2, figsize=(10, 5))

    # sub_ax = inset_axes(
    #     parent_axes=axs,
    #     width="40%",
    #     height="30%",
    #     loc='center right',
    #     borderpad=3  # padding between parent and inset axes
    # )

    # # ideal, noisy, mitigated_linear, mitigated_richardson, mitigated_exp = plotting_choice(taulist, 4, [1, 1.25, 1.5], 0, 0.001, Hlabel="_shots_TI4")
    # ideal, noisy, mitigated, variances, bias, MSE=PLOTTING(taulist, 4, [1, 2, 3], 0, 0.001, Hlabel="_shots_TI4")
    # print(bias)
    # axs.plot(taulist, np.real(ideal), color=colorlist[0], linestyle=linelist[0], )
    # axs.scatter(taulist, noisy,color=colorlist[1], marker="1" )
    # axs.scatter(taulist, mitigated)

    # sub_ax.plot(taulist, np.real(ideal), color=colorlist[0], linestyle=linelist[0], )
    # sub_ax.errorbar(taulist, mitigated, yerr=np.sqrt(np.array(MSE)), capsize=5)
    
    # axs.set_title(r'$p=0.001$')
    # fig.supxlabel(r'simulation time $\tau$')
    # fig.supylabel(r'$\langle I_0\otimes Z_1\otimes Z_2\otimes I_3\rangle$')
    # plt.legend()
    # plt.show()

    fig, axs = plt.subplots(1, figsize=(18, 12), )
    qspdict=PLOTTINGQSP(taulistqsp, 4, [1, 2,3], 0, p, Hlabel="_shots_TI4_fixed_e2",  lchoice='degree', degree=405)
    # qspdict=PLOTTINGQSP(taulistqsp, 4, [1, 2, 3], 0, p, Hlabel="_shots_TI6_fixed_e4",  lchoice='order', eorder=5)

    # axs.plot(taulist, np.real(ideal), color=colorlist[0], linestyle=linelist[0],label='ideal' )
    # axs.scatter(taulist, noisy,color=colorlist[1], marker="1" , label='noisy')
    # axs.scatter(taulist, mitigated, label='mitigated')

    # sub_ax.plot(taulist, np.real(ideal), color=colorlist[0], linestyle=linelist[0], )
    # axs.errorbar(taulist, mitigated, yerr=np.sqrt(np.array(MSE)), capsize=5)
    
    # axs.errorbar(taulist, ideal, yerr=5*10**(-4), capsize=5)
    # axs.plot(taulist, mitigated)
    axs.scatter(taulistqsp, np.real(np.array(qspdict['mitigated'])), label=r"mitigated")
    axs.plot(taulistqsp, np.real(np.array(qspdict['ideal'])), label=r"ideal")
    # axs.plot(taulisttrotter, np.real(np.array(trotterdict['mitigated'])), label=r"ideal")

    # axs.plot(taulistqsp, np.sqrt(np.real(np.array(qspdict['MSE']))), label=r"$\sqrt{MSE}$")
    #axs.plot(taulisttrotter, np.sqrt(np.real(np.array(trotterdict['MSE']))), label=r"$\sqrt{MSE}$")
    # axs.plot(taulist, 3*10**(-4)*np.ones(len(taulist)), label=r"$\epsilon_{QEM}=3*10**(-4)$")
    axs.set_title(r'$p=$'+str(p))
    fig.supxlabel(r'simulation time $\tau$')
    fig.supylabel(r'$\langle I_0\otimes Z_1\otimes Z_2\otimes I_3\rangle$')
    
    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    # tikzplotlib.save(os.path.join(save_path, "expvstaulongsim4qhighp.tex"), flavor="context")

    plt.legend()
    plt.show()

    return 

def APPENDIX_COMP_PLOT(taulist, hlabel="_shots_TI4_shots", ifsave=False, totau=20, d=31):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), )
    
    dict= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.0001, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    axs[0, 0].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[0, 0].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[0, 0].scatter(taulist, dict['mitigated_linear'], color=colorlist[2],marker="2")
    axs[0, 0].scatter(taulist, dict['mitigated_richardson'], color=colorlist[3],  marker=".")
    # axs[0, 0].errorbar(taulist, dict['mitigated_exp'], yerr=np.sqrt(np.array(mitigated_exp)), capsize=5)
    axs[0, 0].scatter(taulist, dict['mitigated_exp'],color=colorlist[4], marker="3")
    axs[0, 0].set_title(r'$p=0.0001$')

    dict= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.001, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    # axs[0, 1].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[0, 1].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[0, 1].scatter(taulist, dict['mitigated_linear'], color=colorlist[2],marker="2")
    axs[0, 1].scatter(taulist, dict['mitigated_richardson'], color=colorlist[3],  marker=".")
    axs[0, 1].scatter(taulist, dict['mitigated_exp'],color=colorlist[4], marker="3")
    axs[0, 1].set_title(r'$p=0.001$')

    dict= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.01, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    axs[0, 2].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[0, 2].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[0, 2].scatter(taulist, dict['mitigated_linear'], color=colorlist[2],marker="2")
    axs[0, 2].scatter(taulist, dict['mitigated_richardson'], color=colorlist[3],  marker=".")
    axs[0, 2].scatter(taulist, dict['mitigated_exp'],color=colorlist[4], marker="3")
    axs[0, 2].set_title(r'$p=0.01$')

    dict= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.0001, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    axs[1, 0].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[1, 0].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[1, 0].scatter(taulist, dict['mitigated_linear'], color=colorlist[2],marker="2")
    axs[1, 0].scatter(taulist, dict['mitigated_richardson'], color=colorlist[3],  marker=".")
    axs[1, 0].scatter(taulist, dict['mitigated_exp'],color=colorlist[4], marker="3")
    axs[1, 0].set_title(r'$p=0.0001$') 
    
    dict= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.001, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    axs[1, 1].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    axs[1, 1].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[1, 1].scatter(taulist, dict['mitigated_linear'], color=colorlist[2],marker="2")
    axs[1, 1].scatter(taulist, dict['mitigated_exp'],color=colorlist[4], marker="3")
    axs[1, 1].set_title(r'$p=0.001$') 

    dict= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.01, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    axs[1, 2].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0],label={'ideal'} )
    axs[1, 2].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" , label={'noisy'})
    axs[1, 2].scatter(taulist, dict['mitigated_linear'], color=colorlist[2],marker="2", label={'linear'})
    axs[1, 2].scatter(taulist, dict['mitigated_richardson'], color=colorlist[3],  marker=".", label={'Richardson'})
    axs[1, 2].scatter(taulist, dict['mitigated_exp'],color=colorlist[4], marker="3", label={'exp'})
    axs[1, 2].set_title(r'$p=0.01$') 

    fig.supxlabel(r'simulation time $\tau$')
    fig.supylabel(r'$\langle I_0\otimes Z_1\otimes Z_2\otimes I_3\rangle$')
    ylabels=[r'scaling $\left[1, 1.25, 1.5\right]$', r'scaling $\left[1, 2, 3\right]$']

    if totau==5:
        savetitle="appendixplotscalintau5.tex"
        fig.suptitle(r'Times $\tau\in\{0.1, 0.1, 0.2,...5\}$')
    else:
        savetitle="appendixplotscalintau20.tex"
        fig.suptitle(r'Times $\tau\in\{5, 5.25, 5.5,...20\}$')
    # for ax in axs.flat:
    #     ax.set( ylabel=ylabels[ax])
    # axs.flat[0].set(ylabel=ylabels[0])
    # axs.flat[1].set(ylabel=ylabels[1])
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()


    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")
    plt.legend()
    plt.show()
    return 

def APPENDIX_COMP_PLOT_MSE(taulist, hlabel="_shots_TI4_shots", ifsave=True, totau=20, d=31):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), )
    
    dict= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.0001, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    # axs[0, 0].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    # axs[0, 0].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[0, 0].scatter(taulist, dict['MSE_lin'], color=colorlist[2],marker="2")
    # axs[0, 0].scatter(taulist, dict['MSE_rid'], color=colorlist[3],  marker=".")
    # axs[0, 0].errorbar(taulist, dict['MSE_exp'], yerr=np.sqrt(np.array(MSE_exp)), capsize=5)
    axs[0, 0].scatter(taulist, dict['MSE_exp'],color=colorlist[4], marker="3")
    axs[0, 0].set_title(r'$p=0.0001$')

    dict= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.001, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    # axs[0, 1].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    # axs[0, 1].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[0, 1].scatter(taulist, dict['MSE_lin'], color=colorlist[2],marker="2")
    # axs[0, 1].scatter(taulist, dict['MSE_rid'], color=colorlist[3],  marker=".")
    axs[0, 1].scatter(taulist, dict['MSE_exp'],color=colorlist[4], marker="3")
    axs[0, 1].set_title(r'$p=0.001$')

    dict= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.01, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    # axs[0, 2].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    # axs[0, 2].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[0, 2].scatter(taulist, dict['MSE_lin'], color=colorlist[2],marker="2")
    # axs[0, 2].scatter(taulist, dict['MSE_rid'], color=colorlist[3],  marker=".")
    axs[0, 2].scatter(taulist, dict['MSE_exp'],color=colorlist[4], marker="3")
    axs[0, 2].set_title(r'$p=0.01$')

    dict= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.0001, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    # axs[1, 0].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    # axs[1, 0].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[1, 0].scatter(taulist, dict['MSE_lin'], color=colorlist[2],marker="2")
    # axs[1, 0].scatter(taulist, dict['MSE_rid'], color=colorlist[3],  marker=".")
    axs[1, 0].scatter(taulist, dict['MSE_exp'],color=colorlist[4], marker="3")
    axs[1, 0].set_title(r'$p=0.0001$') 
    
    dict= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.001, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    # axs[1, 1].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    # axs[1, 1].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[1, 1].scatter(taulist, dict['MSE_lin'], color=colorlist[2],marker="2")
    axs[1, 1].scatter(taulist, dict['MSE_exp'],color=colorlist[4], marker="3")
    axs[1, 1].set_title(r'$p=0.001$') 

    dict= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.01, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    # axs[1, 2].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0],label={'ideal'} )
    # axs[1, 2].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" , label={'noisy'})
    axs[1, 2].scatter(taulist, dict['MSE_lin'], color=colorlist[2],marker="2", label={'linear'})
    # axs[1, 2].scatter(taulist, dict['MSE_rid'], color=colorlist[3],  marker=".", label={'Richardson'})
    axs[1, 2].scatter(taulist, dict['MSE_exp'],color=colorlist[4], marker="3", label={'exp'})
    axs[1, 2].set_title(r'$p=0.01$') 

    fig.supxlabel(r'simulation time $\tau$')
    fig.supylabel(r'$\langle I_0\otimes Z_1\otimes Z_2\otimes I_3\rangle$')
    ylabels=[r'scaling $\left[1, 1.25, 1.5\right]$', r'scaling $\left[1, 2, 3\right]$']

    if totau==5:
        savetitle="appendixplotscalintau5.tex"
        fig.suptitle(r'Times $\tau\in\{0.1, 0.1, 0.2,...5\}$')
    else:
        savetitle="appendixplotscalintau20.tex"
        fig.suptitle(r'Times $\tau\in\{5, 5.25, 5.5,...20\}$')
    # for ax in axs.flat:
    #     ax.set( ylabel=ylabels[ax])
    # axs.flat[0].set(ylabel=ylabels[0])
    # axs.flat[1].set(ylabel=ylabels[1])
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()


    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")
    plt.legend()
    plt.show()
    return 

def MSE_SCALING_COMP(taulist, hlabel="_shots_TI4_shots", ifsave=False, totau=20, d=31):
    fig, axs = plt.subplots(3, 3, figsize=(18, 20))
    
    # dict25= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.0001, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    # dict1= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.0001, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    dict25= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.0001, Hlabel=hlabel, exttrue=True, lchoice='degree', degree=405)
    dict1= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.0001, Hlabel=hlabel, exttrue=True, lchoice='degree', degree=405)
    axs[0, 0].plot(taulist, dict25['MSE_lin'], color=colorlist[0],marker="2", label="[1, 1.25, 1.5]")
    axs[0, 0].plot(taulist, dict1['MSE_lin'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    axs[0, 0].set_title(r'$p=0.0001, lin fit$')
    axs[1, 0].plot(taulist, dict25['MSE_rid'], color=colorlist[0],marker="2", label="[1, 1.25, 1.5]")
    axs[1, 0].plot(taulist, dict1['MSE_rid'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    axs[1, 0].set_title(r'$p=0.0001, rid fit$')
    # axs[0, 0].legend()
    axs[2,0].plot(taulist, dict25['MSE_exp'],color=colorlist[0], marker="2")
    axs[2, 0].plot(taulist, dict1['MSE_exp'],color=colorlist[1], marker="3")
    axs[2, 0].set_title(r'$p=0.0001, exp fit$')

    # dict25= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.001, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    # dict1= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.001, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    dict25= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.001, Hlabel=hlabel, exttrue=True, lchoice='degree', degree=405)
    dict1= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.001, Hlabel=hlabel, exttrue=True, lchoice='degree', degree=405)
    axs[0, 1].plot(taulist, dict25['MSE_lin'], color=colorlist[0],marker="2", label="[1, 1.25, 1.5]")
    axs[0, 1].plot(taulist, dict1['MSE_lin'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    axs[0, 1].set_title(r'$p=0.001, lin fit$')
    axs[1, 1].plot(taulist, dict25['MSE_rid'], color=colorlist[0],marker="2", label="[1, 1.25, 1.5]")
    axs[1, 1].plot(taulist, dict1['MSE_rid'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    axs[1, 1].set_title(r'$p=0.001, rid fit$')
    axs[2,1].plot(taulist, dict25['MSE_exp'],color=colorlist[0], marker="2",  label="[1, 1.25, 1.5]")
    axs[2, 1].plot(taulist, dict1['MSE_exp'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    axs[2, 1].set_title(r'$p=0.001, exp fit$')
    plt.legend()
    # dict25= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.01, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    # dict1= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.01, Hlabel=hlabel, exttrue=True, lchoice='order', eorder=3)
    # axs[0, 2].plot(taulist, dict25['MSE_lin'], color=colorlist[0],marker="2", label="[1, 1.25, 1.5]")
    # axs[0, 2].plot(taulist, dict1['MSE_lin'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    # axs[0, 2].set_title(r'$p=0.01, lin fit$')
    # axs[1, 2].plot(taulist, dict25['MSE_rid'], color=colorlist[0],marker="2", label="[1, 1.25, 1.5]")
    # axs[1, 2].plot(taulist, dict1['MSE_rid'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    # axs[1, 2].set_title(r'$p=0.01, rid fit$')
    # axs[2,2].plot(taulist, dict25['MSE_exp'],color=colorlist[0], marker="2", label="[1, 1.25, 1.5]")
    # axs[2, 2].plot(taulist, dict1['MSE_exp'],color=colorlist[1], marker="3", label="[1, 2, 3]")
    # axs[2, 2].set_title(r'$p=0.01, exp fit$')

    fig.supxlabel(r'simulation time $\tau$')
    fig.supylabel(r'$\langle I_0\otimes Z_1\otimes Z_2\otimes I_3\rangle$')
    # ylabels=[r'scaling $\left[1, 1.25, 1.5\right]$', r'scaling $\left[1, 2, 3\right]$']
    
    savetitle="msescalingcompqsp4.tex"
    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")
    plt.show()
    return 

# MAINTEXTPLOT(taulistlongqsp, p=0.001, scalingqsp=[1, 2, 3], scalingtrotter=[1, 1.25, 1.5])

# MSE_SCALING_COMP(taulistlongqsp, hlabel="_shots_TI4_fixed_e2", ifsave=False)

def IDEALMITCOMPPLOTALLQ(taulist, ifsave=False):
    fig, axs = plt.subplots(3, 3, figsize=(18, 12), )
            
    ###4 qubits
    dict= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.0001, Hlabel="_shots_TI4_fixed_e2", exttrue=True, lchoice='order', eorder=3)
    axs[0, 0].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    # axs[0, 0].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[0, 0].scatter(taulist, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[0, 0].set_title(r'$p=0.0001$')

    dict= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.001, Hlabel="_shots_TI4_fixed_e2", exttrue=True, lchoice='order', eorder=3)
    axs[0, 1].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    # axs[0, 1].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[0, 1].scatter(taulist, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[0, 1].set_title(r'$p=0.001$')

    dict= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.01, Hlabel="_shots_TI4_fixed_e2", exttrue=True, lchoice='order', eorder=3)
    axs[0, 2].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    # axs[0, 2].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[0, 2].scatter(taulist, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[0, 2].set_title(r'$p=0.01$')

    ###6 qubits
    dict= plotting_choice_QSP(taulist, 4, [1, 2, 3], 0, 0.0001, Hlabel="_shots_TI6_fixed_e2", exttrue=True, lchoice='order', eorder=3)
    axs[1, 0].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    # axs[1, 0].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[1, 0].scatter(taulist, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[1, 0].set_title(r'$p=0.0001$') 
    
    dict= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.001, Hlabel="_shots_TI6_fixed_e2", exttrue=True, lchoice='order', eorder=3)
    axs[1, 1].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    # axs[1, 1].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[1, 1].scatter(taulist, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[1, 1].set_title(r'$p=0.001$') 

    dict= plotting_choice_QSP(taulist, 4, [1, 1.25, 1.5], 0, 0.01, Hlabel="_shots_TI6_fixed_e2", exttrue=True, lchoice='order', eorder=3)
    axs[1, 2].plot(taulist, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0],label={'exp'} )
    # axs[1, 2].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" , label={'noisy'})
    axs[1, 2].scatter(taulist, dict['mitigated_exp'], color=colorlist[2],marker="2", label={'exp'})
    axs[1, 2].set_title(r'$p=0.01$') 

     ###8 qubits
    dict= plotting_choice_QSP(taulistp1, 4, [1, 2, 3], 0, 0.0001, Hlabel="_shots_TI8_fixed_e2", exttrue=True, lchoice='order', eorder=3)
    axs[2, 0].plot(taulistp1, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    # axs[1, 0].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[2, 0].scatter(taulistp1, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[2, 0].set_title(r'$p=0.0001$') 
    
    dict= plotting_choice_QSP(taulistp1, 4, [1, 1.25, 1.5], 0, 0.001, Hlabel="_shots_TI8_fixed_e2", exttrue=True, lchoice='order', eorder=3)
    axs[2, 1].plot(taulistp1, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0], )
    # axs[1, 1].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" )
    axs[2, 1].scatter(taulistp1, dict['mitigated_exp'], color=colorlist[2],marker="2")
    axs[2, 1].set_title(r'$p=0.001$') 

    dict= plotting_choice_QSP(taulistp1, 4, [1, 2, 3], 0, 0.01, Hlabel="_shots_TI8_fixed_e2", exttrue=True, lchoice='order', eorder=3)
    axs[2, 2].plot(taulistp1, np.real(dict['ideal']), color=colorlist[0], linestyle=linelist[0],label={'exp'} )
    # axs[1, 2].scatter(taulist, dict['noisy'],color=colorlist[1], marker="1" , label={'noisy'})
    axs[2, 2].scatter(taulistp1, dict['mitigated_richardson'], color=colorlist[2],marker="2", label={'richardson'})
    axs[2, 2].set_title(r'$p=0.01$') 
    fig.supxlabel(r'simulation time $\tau$')
    fig.supylabel(r'$\langle I_0\otimes Z_1\otimes Z_2\otimes I_3\rangle$')

    savetitle="idealmitcompplotallq.tex"
    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")
    plt.show()
    return 

def VARIANCEvsNOISE(taulistqsp, plist, ifsave=False):
    varqsplist4=[]
    varqsplist6=[]
    varqsplist8=[]
    vartrotter=[]
    for t in taulistqsp:
        for p in plist:
            varqsplist4.append(np.real(GETVARS([t], 4, [1], 0, p, Hlabel="_shots_TI4_fixed_e2", lchoice="order")))
            # varqsplist6.append(np.real(plotting_choice_QSP([t], 4, [1, 2,3], 0, p, Hlabel="_shots_TI6_fixed_e2", exttrue=True, lchoice="order")['variances_exp']))

        plt.plot(plist,np.array(varqsplist4), label="4 qubit, "+str(t),  marker="1")
        varqsplist4=[]
        # plt.plot(plist, varsqsplist6, label="6 qubit, "+str(t),  marker="2")
        # varqsp.append(plotting_choice_QSP(taulistqsp, 4, [1, 2,3], 0, p, Hlabel="_shots_TI4_fixed_e2", exttrue=True, lchoice="order")['variances_exp'])
        # varstrotter.append(plotting_choice_QSP(taulistqsp, 4, [1, 2,3], 0, p, Hlabel="TI4_dt01_fixed", exttrue, lchoice="order", eorder=3)['variances_exp'])

    # plt.plot(plist, varqsplist4, label="4 qubits", color=colorlist[0], marker="1")
    # plt.plot(plist, varqsplist6, label="6 qubits", color=colorlist[1], marker="2")
    # savetitle="varvsp"+".tex"
    savetitle="expvsp"+".tex"

    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")
    plt.legend()
    plt.show()
    return

def VARIANCEvsTIME(taulistqsp,taulistlong, plist, ifsave=False):
    varqsplist4=[]
    for p in plist:
        # for t in taulistqsp:
        #     varqsplist4.append(np.real(GETVARS([t], 4, [1], 0, p, Hlabel="_shots_TI4_fixed_e2", lchoice="order", eorder=3)))
            
        for t in taulistlong:
            varqsplist4.append(np.real(GETVARS([t], 4, [1], 0, p, Hlabel="_shots_TI4_fixed_e2", lchoice="degree", degree=405)))
        
        plt.plot(taulistlong,np.array(varqsplist4), label=r"noise is $p=$"+str(p),  marker="1")
        varqsplist4=[]
    
    savetitle="expvslongtime"+".tex"

    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")
    plt.legend()
    plt.show()
    return
# VARIANCEvsNOISE([1.0, 5.0, 15.0],np.arange(1.1e-4, 1.75e-1, 5e-4), ifsave=True)
# VARIANCEvsNOISE([1.0, 5.0, 15.0],np.arange(1.1e-4, 3e-1, 5e-4), ifsave=False)
# VARIANCEvsTIME(taulistp1+taulistp25, taulistlongqsp, [0.0001, 0.001, 0.01], ifsave=True)
def BIASvsTAU(taulistqsp, taulisttrotter, p, scalingqsp=[1, 1.25, 1.5],scalingtrotter=[1, 1.25, 1.5], ifsave=False):
    # biasqsplist=[]
    # biastrotterlist=[]
    biasqsplist=np.real(PLOTTINGQSP(taulistqsp, 4, scalingqsp, 0, p, Hlabel="_shots_TI4_fixed_e2", lchoice="order")['bias'])
    biastrotterlist=np.real(PLOTTING_trotter(taulisttrotter, p, scalingtrotter, Hlabel="TI4_dt01_fixed", lchoice="degree", degree=5)['bias'])

            # varqsplist6.append(np.real(plotting_choice_QSP([t], 4, [1, 2,3], 0, p, Hlabel="_shots_TI6_fixed_e2", exttrue=True, lchoice="order")['variances_exp']))

    plt.plot(taulistqsp,np.array(biasqsplist), label="qsp, noise is "+str(p),  marker="1")
    plt.plot(taulisttrotter,np.array(biastrotterlist), label="trotter, noise is "+str(p),  marker="2")
    plt.plot(taulisttrotter, 0.01*np.ones(len(taulisttrotter)), label="bias threshhold")
    plt.title(r"bias vs tau, noise is $p=$"+str(p))    
    plt.xlabel(r'simulation time $tau$')
    plt.ylabel(r'bias')
    savetitle="biasvstau_p"+str(1000*p)+".tex"
    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")

    plt.legend()
    plt.show()

def BIASvsTAUPRECISE(taulistqsp, p, scalingqsp=[1, 2, 3],scalingtrotter=[1, 2, 3], ifsave=False):
    biasqsplist4=np.real(PLOTTINGQSP(taulistqsp, 4, scalingqsp, 0, p, Hlabel="_shots_TI4_fixed_e4", lchoice="order", eorder=5)['bias'])
    biasqsplist6=np.real(PLOTTINGQSP(taulistqsp, 4, scalingqsp, 0, p, Hlabel="_shots_TI6_fixed_e4", lchoice="order", eorder=5)['bias'])

    plt.plot(taulistqsp,np.array(biasqsplist4), label="$4$ qubits",  marker="1")
    plt.plot(taulistqsp,np.array(biasqsplist6),label="$6$ qubits",  marker="2")
    plt.plot(taulistqsp, 0.0001*np.ones(len(biasqsplist6)), label="bias threshhold")
    plt.title(r"bias vs tau, noise is $p=$"+str(p))    
    plt.xlabel(r'simulation time $tau$')
    plt.ylabel(r'bias')
    savetitle="biasvstau_precise.tex"
    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")

    plt.legend()
    plt.show()

def BIASvsTAULONG(taulistqsp, scalingqsp=[1, 1.25, 1.5],scalingtrotter=[1, 1.25, 1.5], ifsave=False):
    biasqsplist4=np.real(PLOTTINGQSP(taulistqsp, 4, scalingqsp, 0, 0.0001, Hlabel="_shots_TI4_fixed_e2", lchoice='degree', degree=405)['bias'])
    biasqsplist6=np.real(PLOTTINGQSP(taulistqsp, 4, scalingtrotter, 0,0.001, Hlabel="_shots_TI4_fixed_e2", lchoice='degree', degree=405)['bias'])

    plt.plot(taulistqsp,np.abs(np.array(biasqsplist4)), label="$p=0.0001$",  marker="1")
    plt.plot(taulistqsp,np.abs(np.array(biasqsplist6)),label="$p=0.001$",  marker="2")
    plt.plot(taulistqsp, 0.0001*np.ones(len(biasqsplist6)), label="bias threshhold")   
    plt.xlabel(r'simulation time $tau$')
    plt.ylabel(r'bias')
    savetitle="biasvstau_long.tex"
    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")

    plt.legend()
    plt.show()
BIASvsTAULONG(taulistlongqsp, ifsave=True)
# BIASvsTAUPRECISE(taulistp1+taulist5p25,0.0001, scalingqsp=[1, 2,3],scalingtrotter=[1, 2, 3],ifsave=True)
#VARIANCEvsTAU(np.array([1.0, 5.0]),  np.arange(1e-4, 1e-1, 5e-3))
# IDEALMITCOMPPLOTALLQ(taulistqsp, ifsave=True)
# APPENDIX_COMP_PLOT(taulistqsp, hlabel="_shots_TI6_fixed_e2")
def MAINEXPvsTAU(taulistqsp, taulisttrotter, scalingqsp=[1, 1.25,1.5], scalingtrotter=[1, 1.25,1.5], p=0.0001, ifsave=False):
    fig, axs = plt.subplots(1, figsize=(12, 6), )
    qspdict=PLOTTINGQSP(taulistqsp, 4, scalingqsp, 0, p, Hlabel="_shots_TI4_fixed_e2",  lchoice="order")
    trotterdict=PLOTTING_trotter(taulisttrotter, p,scalingtrotter, Hlabel="TI4_dt01_fixed", lchoice="degree", degree=5)
    
    axs.plot(taulistqsp, qspdict['ideal'], color=colorlist[1], label=r"$e^{-i\tau H}$" )
    axs.scatter(taulistqsp, qspdict['mitigated'], color=colorlist[1], marker="1" , label="mitigated QSP")
    # axs.plot(taulisttrotter, trotterdict['ideal'], color=colorlist[2],  label="trotter")
    axs.scatter(taulisttrotter, trotterdict['mitigated'], color=colorlist[2], marker="1" , label="mitigated Trotter")

    # axs.errorbar(taulisttrotter, trotterdict['mitigated'], yerr=np.sqrt(np.array(trotterdict['MSE'])))
    axs.set_ylabel('expectation value')
    axs.set_xlabel(r'simulation time $\tau$')

    savetitle="mainexpvstau"+str(1000*p)+".tex"
    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")
    plt.legend()
    
    plt.show()
    return

def MAINEXPvsTAULONG(taulistqsp, p=0.0001, ifsave=False):
    fig, axs = plt.subplots(1, figsize=(12, 6), )
    qspdict=PLOTTINGQSP(taulistqsp, 4, [1, 1.25,1.5], 0, p, Hlabel="_shots_TI4_fixed_e2",  lchoice="degree", degree=405)
    
    axs.plot(taulistqsp, qspdict['ideal'], color=colorlist[1], label=r"$e^{-i\tau H}$" )
    axs.scatter(taulistqsp, qspdict['mitigated'], color=colorlist[2], marker="1" , label="mitigated QSP")
    # axs.plot(taulisttrotter, trotterdict['ideal'], color=colorlist[2],  label="trotter")

    # axs.errorbar(taulisttrotter, trotterdict['mitigated'], yerr=np.sqrt(np.array(trotterdict['MSE'])))
    axs.set_ylabel('expectation value')
    axs.set_xlabel(r'simulation time $\tau$')

    savetitle="mainexpvstaulong"+str(1000*p)+".tex"
    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")
    plt.legend()
    
    plt.show()
    return

def MAINEXPvsTAUPRECISE(taulistqsp, Hlabel="_shots_TI4_fixed_e4", p=0.0001, ifsave=False):
    fig, axs = plt.subplots(1, figsize=(12, 6), )
    qspdict=PLOTTINGQSP(taulistqsp, 4, [1, 2, 3], 0, p, Hlabel, lchoice="order", eorder=5)
    
    axs.plot(taulistqsp, qspdict['ideal'], color=colorlist[1], label=r"$e^{-i\tau H}$" )
    axs.scatter(taulistqsp, qspdict['mitigated'], color=colorlist[2], marker="1" , label="mitigated QSP")
    # axs.plot(taulisttrotter, trotterdict['ideal'], color=colorlist[2],  label="trotter")
    # axs.scatter(taulisttrotter, trotterdict['mitigated'], color=colorlist[2], marker="1" , label="mitigated Trotter")

    # axs.errorbar(taulisttrotter, trotterdict['mitigated'], yerr=np.sqrt(np.array(trotterdict['MSE'])))
    axs.set_ylabel('expectation value')
    axs.set_xlabel(r'simulation time $\tau$')

    savetitle="mainexpvstauprecise6q.tex"
    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")
    plt.legend()
    
    plt.show()
    return
# MAINEXPvsTAU(taulistqsp, taulisttrotter,p=0.01, scalingqsp=[1, 2,3],scalingtrotter=[1, 2, 3],ifsave=False)4
# MAINEXPvsTAUPRECISE(taulistp1+taulist5p25, "_shots_TI6_fixed_e4", ifsave=False)
# MAINEXPvsTAULONG(taulistlongqsp, p=0.001, ifsave=False)

def FIXEDvsBOUNDSHOTS(taulistqsp,taulisttrotter, p=0.0001, ifsave=False):
    fixedtrotter=PLOTTING_trotter(taulisttrotter, p, [1, 2,3],  Hlabel="TI4_dt01_fixed",  lchoice="degree", degree=5)
    boundtrotter=PLOTTING_trotter(taulisttrotter, p, [1, 2,3],  Hlabel="TI4_dt01",  lchoice="degree", degree=5)
    fixedqsp=PLOTTINGQSP(taulistqsp, 4, [1, 2,3], 0, p, Hlabel="_shots_TI4_fixed_e2",  lchoice="order")
    boundqsp=PLOTTINGQSP(taulistqsp, 4, [1, 2,3], 0, p, Hlabel="_shots_TI4_e2",  lchoice="order")
    plt.plot(taulistqsp, fixedqsp['mitigated'], colorlist[0], label="QSP fixed")
    plt.plot(taulistqsp, fixedqsp['ideal'], colorlist[2], label="actual")
    
    plt.scatter(taulistqsp, boundqsp['mitigated'],  label="QSP bound")
    plt.plot(taulisttrotter, fixedtrotter['mitigated'],  label="trotter fixed")
    plt.scatter(taulisttrotter, boundtrotter['mitigated'],  label="trotter bound")
    plt.xlabel(r'simulation time $tau$')
    plt.ylabel(r'expectation value')
    plt.title(r'expectation value vs time, $p=$'+str(p))
    savetitle="fixedboundcompq4"+str(1000*p)+".tex"
    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    if ifsave==True:
        tikzplotlib.save(os.path.join(save_path, savetitle), flavor="context")

    plt.legend()
    plt.show()
    return 

# FIXEDvsBOUNDSHOTS(taulistqsp, taulisttrotter, ifsave=True, p=0.0001) 

def IDEALMITCOMPALLQ(taulistqsp, p=0.001):
    fig, axs = plt.subplots(1, figsize=(6, 8), )
    qspdict4=PLOTTINGQSP(taulistqsp, 4, [1, 2,3], 0, p, Hlabel="_shots_TI4_fixed_e2",  lchoice="order")
    qspdict6=PLOTTINGQSP(taulistqsp, 4, [1, 2,3], 0, p, Hlabel="_shots_TI6_fixed_e2",  lchoice="order")
    # qspdict8=PLOTTINGQSP(taulistqsp, 4, [1, 2,3], 0, p, Hlabel="_shots_TI8_fixed_e2",  lchoice="order")

    axs.plot(taulistqsp, qspdict4['ideal'], color=colorlist[1], label="4 qubits")
    axs.scatter(taulistqsp, qspdict4['mitigated'], color=colorlist[1], marker="1" )

    axs.plot(taulistqsp, qspdict6['ideal'], color=colorlist[2], label="6 qubits" )
    axs.scatter(taulistqsp, qspdict6['mitigated'], color=colorlist[2], marker="1" )

    # axs.plot(taulistqsp, qspdict8['ideal'], color=colorlist[1], marker="1" )
    # axs.scatter(taulistqsp, qspdict8['mitigated'], color=colorlist[1], marker="1" )
    # axs.plot(taulisttrotter, trotterdict['ideal'], color=colorlist[2], marker="1" )
    # axs.scatter(taulisttrotter, trotterdict['MSE'], color=colorlist[2], marker="1" )
    plt.legend()
    plt.show()
    return
# IDEALMITCOMPALLQ(taulistqsp)
# MAINEXPvsTAU(taulistqsp, taulisttrotter)
def plottingnoises(tau, prec, scaling, l, noises, Hlabel="SK4", exttrue=True, lchoice="degree", degree=13, eorder=3):

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
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_deg_" +str(degree) + "_t_" + str(tau) 
        elif lchoice=='order':
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_eorder_" + str(eorder) + "_t_" + str(tau) 
        if prec>=5:
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" + "1.0e-"+str(prec) + "_t_" + str(tau)
        else:
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" +str(10**(-prec)) + "_t_" + str(tau)
        current_path = os.path.abspath(__file__)
        parentdir=os.path.dirname(current_path)
        datadir=os.path.join(parentdir, "benchmark_data")
        with open(os.path.join(datadir, filename+'.pkl'), 'rb') as f:
            datadict = pickle.load(f)
        nois = datadict['noisyexpdict_'+str(l)]["noisyexpectationarray_"+str(1)]
        idi=datadict['classicalvals'][l]
        noisy.append(nois)
        ideal.append(idi)
        variance = datadict['noisyexpdict_'+str(l)]['variance']
        variance = [variance[i]/5e+6 for i in indices]
        #variance = [variance[i] for i in indices]
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
    

    # print(variances_lin)
    # print(variances_rid)
    # print(variances_exp)

    if exttrue==True:
        return ideal, noisy, mitigated_linear, mitigated_richardson, mitigated_exp#, variances_lin, bias_lin, MSE_lin, mitigated_exp, variances_exp, bias_exp, MSE_exp, mitigated_richardson, variances_rid, bias_rid, MSE_rid
    elif exttrue=='vars':
        return  ideal, noisy, variances_lin, variances_rid
    else:
        return ideal, noisy, mitigated_linear, mitigated_richardson

def EXPvsNOISEPLOT():
    noisearray=np.linspace(0.001, 0.3, 30)
    # noisearray=[0.0015, 0.0025, 0.003, 0.0035, 0.004, 0.0055, 0.006]
    
    fig, axs = plt.subplots(1, figsize=(18, 12), )
    fig.supxlabel(r'noise parameter $p$')
    fig.supylabel(r'$\langle I_0\otimes Z_1\otimes Z_2\otimes I_3\rangle$')
    ylabels=[r'scaling $\left[1, 1.25, 1.5\right]$', r'scaling $\left[1, 2, 3\right]$']
    # fig.suptitle(r'Times $\tau\in\{1, 1.1, 1.2,...5\}$')

    # axs.flat[0].set(ylabel=ylabels[0])
    # axs.flat[1].set(ylabel=ylabels[1])
    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()
    taulist=[1, 5, 15]
    for j in range(3):
        ideal, noisy, mitigated_linear, mitigated_richardson=plottingnoises(taulist[j], 4, [1, 1.25, 1.5], 1, noisearray, Hlabel="TI4", exttrue=False)
        axs.plot(noisearray,mitigated_richardson, color=colorlist[j], linestyle=linelist[j], label=str(taulist[j]))
        

    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    # tikzplotlib.save(os.path.join(save_path, "varvsnoise.tex"), flavor="context")
    plt.legend()
    plt.show()

# plots for M vs D
def MvsDPLOTS():
    D = np.arange(1, 1000, 1)
    deltas = [1e-4, 1e-6,  1e-12]

    n=4
    l = 0.001
    full = []
    for delta in deltas:
        epsilon = 3*delta
        Ms = []
        for d in D:
            Mup = np.log10(2/epsilon)
            Mdown = (1-l)**d*4*np.log(2)*n*2*delta**2
            M = Mup/Mdown
            M = int(M)
            Ms.append(M)
        full.append(Ms)



    logs2=[math.log10(full[2][j]) for j in range(len(full[2]))]

    l = 0.01
    full2 = []
    for delta in deltas:
        epsilon = 3*delta
        Ms = []
        for d in D:
            Mup = np.log10(2/epsilon)
            Mdown = (1-l)**d*4*np.log(2)*n*2*delta**2
            M = Mup/Mdown
            M = int(M)
            Ms.append(M)
        full2.append(Ms)



    logs22=[math.log10(full2[2][j]) for j in range(len(full[2]))]

    fig, ax = plt.subplots(1 , 2, figsize=(10, 5))

    # main plot
    # ax.plot(D, (np.array(full[0])))
    # ax.plot(D, (np.array(full[1])))
    # ax.plot(D, full[2])

    # create inset plot
    # sub_ax = inset_axes(
    #     parent_axes=ax,
    #     width="40%",
    #     height="30%",
    #     loc='lower right',
    #     borderpad=3  # padding between parent and inset axes
    # )

    # ax[0].set(ylabel=r"Shot bound $M$")

    ax[0].plot(D, np.log10(np.array(full[0])))
    ax[0].plot(D, np.log10(np.array(full[1])))
    ax[0].plot(D, logs2)

    ax[1].plot(D, np.log10(np.array(full2[0])))
    ax[1].plot(D, np.log10(np.array(full2[1])))
    ax[1].plot(D, logs22)

    # sub_ax.set(ylabel=r"$\log_{10}(M)$")
    
    fig.supylabel(r'$\log_{10}(M)$')
    fig.supxlabel(r'simulation time $\tau$')

    pathname="plotting.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    tikzplotlib.save(os.path.join(save_path, "shotsvstausub.tex"), flavor="context")
    plt.show()
    return

def DEPTHvsTAU(taulist, prec=4, pU=0.001, dor=1, Hlabel="TI4", lchoice="degree", degree=13):

    # do mitigation and generate mitigated and noisy values for the plots
    degreelist=[]

    for tau in taulist:
        pathname="plotting.py"
        if lchoice=='degree':
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_deg_" +str(degree) + "_t_" + str(tau) 
        elif lchoice=='order':
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_eorder_" + str(5) + "_t_" + str(tau) 
        elif prec>=5:
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" + "1.0e-"+str(prec) + "_t_" + str(tau)
        else:
            filename="noisymmnts"+Hlabel+"p="+str(np.int64(10000*pU)) +"_epsi_" +str(10**(-prec)) + "_t_" + str(tau)
        current_path = os.path.abspath(__file__)
        parentdir=os.path.dirname(current_path)
        datadir=os.path.join(parentdir, "benchmark_data")
        with open(os.path.join(datadir, filename+'.pkl'), 'rb') as f:
            datadict = pickle.load(f)
        
        ###due to general dumbassery in data saving, recalculate the classical values###
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
    # tikzplotlib.save(os.path.join(save_path, "depthvstau2.tex"), flavor="context")
    
    plt.legend()
    plt.show()
    return 

# DEPTHvsTAU(taulist5, degree=13, Hlabel="_shots_TI6_fixed")
# MvsDPLOTS()
# EXPvsNOISEPLOT()
#def NSHOTSvsDEPTH():
    