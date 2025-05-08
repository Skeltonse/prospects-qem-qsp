import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import os
import simulators.matrix_fcns as mf
import functions.laur_poly_fcns as lpf
import parameter_finder as pf
import solvers.Wilson_method as wm
from simulators.projector_calcs import Ep_PLOT, SU2_CHECK, UNIFY_PLIST, BUILD_PLIST
import csv
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

def GETVALS(tau, degree=1, lchoice='order', orderval=5):
    subnorm=1/np.sqrt(2)
    lchoice=lchoice
    if lchoice=='degree':
        filename="hsim_coeffs_deg_" +str(degree) + "_t_" + str(tau) 
    elif lchoice=='order':
        filename="hsim_coeffs_eorder_" + str(orderval) + "_t_" + str(tau) 
    elif precision>=5:
        filename="hsim_coeffs_epsi_" + "1.0e-"+str(approxprecision) + "_t_" + str(tau)
    else:
        filename="hsim_coeffs_epsi_" +str(10**(-approxprecision)) + "_t_" + str(tau)
        
    
        
    current_path = os.path.abspath(__file__)
    parentdir=os.path.dirname(current_path)
    parentdir=parentdir
    datadir=os.path.join(parentdir, "csv_files")
        
    abs_path=os.path.join(datadir, filename+".csv")
        
    # os.makedirs(abs_path, exist_ok=True)
    with open(abs_path, 'r') as file:
        csv_reader = csv.reader(file)
        column1=[]
        column2=[]
        column3=[]
        column4=[]
        next(csv_reader)

        for row in csv_reader:
            col1, col2, col3, col4= row[:4]  # Unpack the first three columns
            column1.append(col1)
            column2.append(col2)
            column3.append(col3)
                
            column4.append(col4)
        ###I keep imaginary numbers in the coeff arrays so each array will produce a real-on-circle poly without pain
        ###GET RID OF ANY INCONVENIANT SMALL COEFFICIENTS###
    a=np.array(column1).astype(float)
    b=np.array(column2).astype(float)
        
    n=int(column3[0])
    epsicoeff=float(column4[0])
    coeffcutoff=10**(-16)
    while abs(a[0])<coeffcutoff and abs(b[0])<coeffcutoff:
        a=a[1:2*n]
        b=b[1:2*n]
        n=n-1
    a=subnorm*(2)*a
    b=subnorm*(2)*b
    n=n
        

    ###getting a QSP circuit
    inst_tol=10**(-14)
    theta=np.linspace(0,np.pi, 1000)
    cz2list=lpf.LAUR_POLY_MULT(a, a)
    sz2list=lpf.LAUR_POLY_MULT(b, b)
    add1=np.append(np.append(np.zeros(2*n), 1), np.zeros(2*n))
    abunc=cz2list+sz2list
    abun=lpf.LAUR_POLY_BUILD(abunc, 2*n,  np.exp(1j*theta))
    calFc=add1-abunc
    gammaW, itW, initgamma, nu, Ttilde=wm.WILSON_LOOP_WCHECK(np.real(calFc), 2*n, nu=inst_tol, init="Wilson", datatype='float')
    c, d, probflag=pf.cd_PROCESS(gammaW, a, b,n, tol=inst_tol,plots=False)
    Plist, Qlist, E0=UNIFY_PLIST(a, -b, c, d, n, inst_tol)
    fcnvals=subnorm*np.exp(-1j*tau*np.cos(theta))
    epsiqsp=pf.NORM_EXTRACT_FROMP(Plist, Qlist, E0,a, b, n, fcnvals, theta)

    return epsicoeff, epsiqsp, n, tau

def GETVALS_ABBREV(tau, degree=1, lchoice='order', orderval=5):
    subnorm=1/np.sqrt(2)
    lchoice=lchoice
    if lchoice=='degree':
        filename="hsim_coeffs_deg_" +str(degree) + "_t_" + str(tau) 
    elif lchoice=='order':
        filename="hsim_coeffs_eorder_" + str(orderval) + "_t_" + str(tau) 
    elif precision>=5:
        filename="hsim_coeffs_epsi_" + "1.0e-"+str(approxprecision) + "_t_" + str(tau)
    else:
        filename="hsim_coeffs_epsi_" +str(10**(-approxprecision)) + "_t_" + str(tau)
        
    
        
    current_path = os.path.abspath(__file__)
    parentdir=os.path.dirname(current_path)
    parentdir=parentdir
    datadir=os.path.join(parentdir, "csv_files")
        
    abs_path=os.path.join(datadir, filename+".csv")
        
    # os.makedirs(abs_path, exist_ok=True)
    with open(abs_path, 'r') as file:
        csv_reader = csv.reader(file)
        column1=[]
        column2=[]
        column3=[]
        column4=[]
        next(csv_reader)

        for row in csv_reader:
            col1, col2, col3, col4= row[:4]  # Unpack the first three columns
            column1.append(col1)
            column2.append(col2)
            column3.append(col3)
                
            column4.append(col4)
        ###I keep imaginary numbers in the coeff arrays so each array will produce a real-on-circle poly without pain
        ###GET RID OF ANY INCONVENIANT SMALL COEFFICIENTS###
    a=np.array(column1).astype(float)
    b=np.array(column2).astype(float)
        
    n=int(column3[0])
    epsicoeff=float(column4[0])
    coeffcutoff=10**(-16)
    while abs(a[0])<coeffcutoff and abs(b[0])<coeffcutoff:
        a=a[1:2*n]
        b=b[1:2*n]
        n=n-1
    a=subnorm*(2)*a
    b=subnorm*(2)*b
    n=n
    
    return epsicoeff, n, tau

def CIRCUIT_GROWTH():
    taulist1=np.around(np.linspace(0.1, 5, 50), 1).tolist()
    
    taulist2= [5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0]
    taulist=taulist1+taulist2
    taulist3=taulist
    nlist=np.zeros(len(taulist))
    dlist=np.zeros(len(taulist))

    nlist2=np.zeros(len(taulist3))
    dlist2=np.zeros(len(taulist3))

    for tind, t in enumerate(taulist):
        epsicoeff,nlist[tind], tau=GETVALS_ABBREV(t, lchoice='order')
        dlist[tind]=2*nlist[tind]+1

    for tind, t in enumerate(taulist3):
        epsicoeff2,nlist2[tind], tau=GETVALS_ABBREV(t, lchoice='order',orderval=3)
        dlist2[tind]=2*nlist2[tind]+1
    
    plt.plot(taulist, nlist, marker='o', label=r'degree $n$, , $\epsilon_{QSP}=10^{-4}$')
    plt.plot(taulist, dlist, label=r'depth $D$, , $\epsilon_{QSP}=10^{-4}$')

    plt.plot(taulist3, nlist2, marker='o', label=r'degree $n$, , $\epsilon_{QSP}=10^{-2}$')
    plt.plot(taulist3, dlist2, label=r'depth $D$, $\epsilon_{QSP}=10^{-2}$')

    plt.ylabel(r"Growth of QSP circuit")
    plt.xlabel(r"Simulation time $\tau$")
    
    pathname="prelim_plots.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    tikzplotlib.save(os.path.join(save_path, "depthvstau.tex"), flavor="context")
    plt.legend()
    plt.show()

    return


def TAUPOSSvsDEPTH():
    taulist1=np.around(np.linspace(0.1, 5, 50), 1).tolist()
    taulist2= [5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0]
    taulist=taulist1+taulist2
    
    taulist3=taulist
    bestnlist=np.zeros(len(taulist))
    bestdlist=np.zeros(len(taulist))
    rlimitlist=np.zeros(len(taulist))

    bestnlist2=np.zeros(len(taulist3))
    bestdlist2=np.zeros(len(taulist3))
    rlimitlist2=np.zeros(len(taulist3))

    degreeoptions=[5, 7, 9, 11, 13, 15, 17, 19,21, 23, 25, 27, 29, 31]
    j=0
    for tind, t in enumerate(taulist):
        while bestnlist[tind]==0:
            epsicoeff, epsiqsp, n, tau=GETVALS(t, degree=degreeoptions[j] ,lchoice='degree')
            
            if epsiqsp<1*10**(-4):
                bestnlist[tind]=n
                bestdlist[tind]=2*n+1
                rlimitlist[tind]=2*np.ceil(4*np.log(1/epsicoeff)/np.log(np.exp(1)+np.log(1/epsicoeff)/t))+1
                break
            
            if degreeoptions[j]==degreeoptions[-1]:
                print("no such degree")
                break
            j+=1
    j=0
    for tind, t in enumerate(taulist3):
        while bestnlist2[tind]==0:
            epsicoeff, epsiqsp, n, tau=GETVALS(t, degree=degreeoptions[j] ,lchoice='degree')
            
            if epsiqsp<1*10**(-2):
                bestnlist2[tind]=n
                bestdlist2[tind]=2*n+1
                rlimitlist2[tind]=2*np.ceil(4*np.log(1/epsicoeff)/np.log(np.exp(1)+np.log(1/epsicoeff)/t))+1
                break
            
            if degreeoptions[j]==degreeoptions[-1]:
                print("no such degree")
                break
            j+=1

    
    plt.plot(taulist,rlimitlist,  label=r"$\tilde{r}\left(\tau, 10^{-4}\right))$")
    plt.plot(taulist, bestnlist, marker='o', label=r'degree $n$')

    plt.plot(taulist3,rlimitlist2,  label=r"$\tilde{r}\left(\tau, 10^{-2}\right))$")
    plt.plot(taulist3, bestnlist2, marker='o', label=r'degree $n_2$')
    plt.plot()
    print(bestnlist2)
    print(np.average(rlimitlist/bestnlist))
    print(np.average(rlimitlist2/bestnlist2))
    plt.ylabel(r"Growth of QSP circuit")
    plt.xlabel(r"Simulation time $\tau$")
    
    pathname="prelim_plots.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    tikzplotlib.save(os.path.join(save_path, "neccdegvstau20.tex"), flavor="context")
    plt.legend()
    plt.show()
    
    return
def GROWTH_OF_EPSI():
    taulist1= [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9, 2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9, 3.1, 3.2, 3.3, 3.4,3.6, 3.7, 3.8, 3.9, 4.1, 4.2, 4.3, 4.4, 4.6, 4.7, 4.8, 4.9]
    taulist2 = [5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0]

    taulist=np.concatenate((taulist1, taulist2))

    taulist3= [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0]

    epsicoefflist=np.zeros(len(taulist))
    epsiqsplist=np.zeros(len(taulist))
    epsicoefflist2=np.zeros(len(taulist))
    epsiqsplist2=np.zeros(len(taulist))
    nlist=np.zeros(len(taulist))
    nlist2=np.zeros(len(taulist))

    for tind, t in enumerate(taulist):
        epsicoefflist[tind], epsiqsplist[tind], nlist[tind], tau=GETVALS(t, lchoice='order')
    
 
    for tind, t in enumerate(taulist3):
        epsicoefflist2[tind], epsiqsplist2[tind], nlist[tind], tau=GETVALS(t, lchoice='order', orderval=3)
    
    plt.plot(-np.log10(1/np.concatenate((epsicoefflist, epsicoefflist2))), -np.log10(1/np.concatenate((epsiqsplist, epsiqsplist2))), marker='o', label=r'$\epsilon_{qsp}$')

    plt.ylabel(r"Error in QSP circuit $\epsilon_{QSP}$")
    plt.xlabel(r"Coefficient error $\epsilon_{coeff}$")
    pathname="prelim_plots.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    # tikzplotlib.save(os.path.join(save_path, "epsivsepsi.tex"), flavor="context")
    plt.legend()
    plt.show()
    print(max(epsiqsplist), max(epsiqsplist2))
    return


def FIXEDDEPTHS(depthlist):
    taulist1=np.around(np.linspace(0.1, 5, 50), 1).tolist()
    taulist2 = [5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0]
    taulist=np.append(taulist1, taulist2)
    epsilist=np.zeros(len(taulist))
    fig, axs = plt.subplots()
    for d in depthlist:
        epsilist=np.array([GETVALS(taulist[j], degree=d, lchoice='degree')[1] for j in range(len(taulist))])
        axs.plot(taulist, -np.log10(1/epsilist), marker='o', label=r"d="+str(d))
    axs.set_xlabel(r'simulation time $\tau$')
    axs.set_ylabel(r'$\log\left(\epsilon_{QSP}\right)$')

    pathname="prelim_plots.py"
    current_path=os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"figures")
    save_path = os.path.normpath(save_path)
    tikzplotlib.save(os.path.join(save_path, "fixeddepthsteps.tex"), flavor="context")
    plt.legend()
    plt.show()
    return
