"""
Functions to process, compute, or check Laurent polynomial lists
Also includes functions useful for plotting data
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time 

'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{braket}')
#matplotlib.verbose.level = 'debug-annoying'


'''LOCAL IMPORTS'''
import functions.laur_poly_fcns as lpf 
import solvers.Wilson_method as wm
from simulators.projector_calcs import Ep_PLOT, SU2_CHECK, UNIFY_PLIST, BUILD_PLIST
import simulators.matrix_fcns as mf


###FUNCTIONS TO CHECK POLYNOMIAL CONDITIONS OR PROCESS COEFFICIENTS
def ab_PROCESS(a, b, n,  theta=np.linspace(-np.pi,np.pi, 100), tol=10**(-16), plots=True, ax=None, **plt_kwargs):
    """
    builds the coefficient array for F(z)=1-a^2(z)-b^2(z) from coeff arrays a, b real valued Laurent polys.
    Checks if F is real-on-circle within tolerance and prints warning if not
    (optional) plots F 

    inputs:
    a, b: 2n+1 length np arrays storing coefficient lists
    n: float, degree of polynomial
    theta: list of points on [-pi, pi]
    tol: tolerance for error
    plots: True/False determines whether to generate a plot
    ax, **plt_kwargs: optional argements for plotting

    returns:
    a, b: same as inputs
    calcFc: length 4n+1 np array with coefficients of the Fejer input polynomial $1-a^2-b^2$
    calF: calF: np array with values of the Fejer input polynomial for each value in \theta
    """
    ###BUILD LIST FOR THE FEJER INPUT POLYNOMIAL, $1-a^2-b^2$###
    cz2list=lpf.LAUR_POLY_MULT(a, a)
    sz2list=lpf.LAUR_POLY_MULT(b, b)
    add1=np.append(np.append(np.zeros(2*n), 1), np.zeros(2*n))
    abunc=cz2list+sz2list
    abun=lpf.LAUR_POLY_BUILD(abunc, 2*n,  np.exp(1j*theta))
    calFc=add1-abunc

    ###CHECK THAT THE ANSWER IS REAL###
    calF=lpf.REAL_CHECK(calFc, 2*n, theta=theta, tol=tol)

    if plots==True:
        if ax is None:
            ax = plt.gca()
        ax.plot(theta, np.real(calF),label=r"$1-a^2(z)-b^2(z)$", **plt_kwargs)
        ax.set_title(r'Plots for Fejer Prob Input Poly')

    
    return a, b, calFc, calF

def cd_PROCESS(gamma, a, b, n, theta=np.linspace(-np.pi,np.pi, 100), tol=10**(-16), plots=True, ax=None, **plt_kwargs):
    """
    builds the coefficient lists for c, d

    inputs:
    gamma: 2n+1 length np array, the coefficients of the solution to the fejer problem
    a, b: 2n+1 length np arrays storing coefficient lists
    n: float, degree of polynomial
    theta: list of points on [-pi, pi]
    tol: tolerance for error
    plots: True/False determines whether to generate a plot
    ax, **plt_kwargs: optional argements for plotting

    returns:
    c, d: 2n+1 coefficient lists representing the real and imaginary parts of the Fejer solution
    probflag: binary variable, 0 is default and 1 signals that there is a problem with the solution
    """
    probflag=0

    ###GET c, d AS REAL-ON-CIRCLE POLYNOMIALS AND CHECK PROPERTIES###
    c=(gamma+np.flip(gamma))/2
    d=-1j*(gamma-np.flip(gamma))/2
    lpf.REAL_CHECK(c, n, theta, tol, 'c')
    lpf.REAL_CHECK(d, n,  theta, tol, 'd')

    ###CHECK THE SUM OF SQUARED a, b, c, d IS ALWAYS 1###
    Fcheckc=lpf.LAUR_POLY_MULT(a, a)+lpf.LAUR_POLY_MULT(b, b)+lpf.LAUR_POLY_MULT(c, c)+lpf.LAUR_POLY_MULT(d, d)
    Fcheck=lpf.LAUR_POLY_BUILD(Fcheckc, 2*n,  np.exp(1j*theta))
    if plots==True:
        if ax is None:
            ax = plt.gca()
        ax.plot(theta, np.real(Fcheck), label=r'$a^2(z)+b^2(z)+c^2(z)+d^2(z)$')
        ax.plot(theta, np.real(lpf.LAUR_POLY_BUILD(c, n, np.exp(1j*theta))), label='c')
        ax.plot(theta, np.real(lpf.LAUR_POLY_BUILD(d, n, np.exp(1j*theta))), label='d')
        ax.legend()
    else:
        problems=np.where(abs(Fcheck-1)>tol)
        if problems[0]!=[]:
            print('probelm, a, b, c, do not obey constraint')
            print(problems)
            probflag=1
    return c, d, probflag


def F_CHECK(calFc, n, theta=np.linspace(-np.pi,np.pi, 100), tol=10**(-16),  rtn='none', fcnname='F'):
    """
    checks if Laurent poly with coefficients calFc is real and positive on the unit circle

    inputs:
    calFc: length 2n+1 coefficient list
    n: float, degree of the Laurent polynomial
    theta: list of points on [-pi, pi]
    tol: tolerance for error
    rtn: option to return the array of the polynomial evalauted along points in \theta
    fcnname: string labelling the polynomial being evaluated    
    """
    calF=lpf.LAUR_POLY_BUILD(calFc, n, np.exp(theta*1j))
    if any(abs(np.imag(calF))>tol):
        print(r'Warning'+fcnname+ 'has imaginary terms')
        print(max(abs(np.imag(calF))))
    elif any(np.real(calF)<=0):
        print(r'Warning, '+ fcnname + ' has negative real terms')
        print(min((np.real(calF))))
    if rtn=='fcn_vals':
        return calF
    else:
        return

def GAMMA_PROCESSING(gamma, n, calF, theta=np.linspace(-np.pi,np.pi, 100), tol=10**(-16),plots=False, ax=None, **plt_kwargs):
    """
    Computes a normalization factor for the Fejer solution. This should be unnecessary for the Wilson method
    
    inputs:
    gamma: 2n+1 length np array, the coefficients of the solution to the fejer problem
    n: float, degree of polynomial
    calF: 4n+1 length np array, the coefficients of Fejer input 
    theta: list of points on [-pi, pi]
    tol: tolerance for error
    plots: True/False determines whether to generate a plot
    ax, **plt_kwargs: optional argements for plotting

    returns:
    the average subnormalization needed to make the solution work 
    """
    coeffp=lpf.LAUR_POLY_MULT(gamma, np.flip(gamma))
    ###generates values over $\theta\in[-\pi, \pi]$ and checks for problems
    calFp=F_check(coeffp, 2*n, tol, theta, rtn='fcn_vals')
    if ax is None:
        ax = plt.gca()

    alpha_list=np.real((calF/calFp))  
    
    if plots==True:
        #ax.plot(theta, np.real(calF), label='og function to solve', **plt_kwargs)
        ax.plot(theta, np.real(calFp), label=r'$\gamma(z)\gamma(1/z)$', **plt_kwargs)
        ax.plot(theta, np.real(calFp)*np.mean(alpha_list), label='normalized solution', **plt_kwargs)
        ax.legend()
        ax.set_title(r"Compare Wilson solution guess to $1-|\mathcal{A}|^2$")
    #return np.real(calFp)*np.mean(alpha_list), np.mean(alpha_list)
    return np.mean(alpha_list)
              

####FIND THE QSP CIRCUIT###
'''BEGIN FINDING THE SOLUTION: '''
def PARAMETER_FIND(czlist, szlist,n, data,epsi=10**(-14), defconv='ad', ifdecomp=True, tDict={}, plots=False, axeschoice=[0, 1]):
    """
    Runs the Fejer solver for each instance. checks incoming polynomial lists, builds Feer input $\mathcal{F}(z)$,
    solves for solution, check sit,  computes $c(z), d(z)$ and checks them.
    computes projectors defining the QSP circuit
    (optional) displays plots for each step and/or times the completion step 

    inputs:
    czlist, szlist: 2n+1 length np arrays storing coefficient lists
    n: float, degree of polynomial
    data: np array with a list of points
    epsi: tolerance for error
    defconv: determines whether d is defined with the reciprocal or the anti-reciprocal part
    ifdecomp: True/False determines whether to complete the decomposition step
    tdict: defines a dictionary to store the solution
    plots: True/False determines whether to generate a plot
    axeschoice, **plt_kwargs: optional argements for plotting
    tdict:
    
    returns:
    a, b: same as inputs
    calcFc: length 4n+1 np array with coefficients of the Fejer input polynomial $1-a^2-b^2$
    calF: calF: np array with values of the Fejer input polynomial for each value in \theta
    """
    ###BOUNDED ERROR ONLY: DEFIN THE PRECISION OF THE FEJER STEP###
    #epsifejer=epsi/(2*n**2+5*n+1)/128
    epsifejer=epsi


    ###CHECK UNPUT POLYNOMIALS AND SOLVE THE COMPLETION STEP###
    t0=time.perf_counter()
    a, b, calFc, calF=ab_PROCESS(czlist, szlist, n,  theta=data, tol=epsi, plots=False,  ax=axeschoice[0])
    gammaW, itW, initgamma, nu=wm.WILSON_LOOP_WDATA(np.real(calFc), 2*n, nu=epsifejer, init="Wilson", datatype='float')
    c, d, probflag=cd_PROCESS(gammaW, a, b,n, tol=epsi,plots=plots, ax=axeschoice[1])
    t1=time.perf_counter()

    ###IF THE SOLUTION FAILS, TRY ANOTHER INITIAL GUESS###
    if probflag==1:
        print('need to try another guess')
        a, b, calFc, calF=ab_PROCESS(czlist, szlist, n, theta=data, tol=epsi , plots=plots,  ax=axeschoice[1] )
        gammaW, itW, initgamma, nu=wm.WILSON_LOOP_WDATA(np.real(calFc), 2*n,  nu=epsifejer, init="Wilson2", datatype='float')
        c, d, probflag=cd_PROCESS(gammaW, a, b,n, tol=epsi, plots=plots, ax=axeschoice[1])
        t1=time.perf_counter()
       
    print('time to find solution', t1-t0)
    solDict={'soltime': t1-t0, 'solit':itW,'degree':n ,'a':a, 'b':b,  'c': c, 'd': d, 'gamma': gammaW, 'initialguess':initgamma, 'wilsontol': nu, 'rerunflag':probflag}
    tDict.update(solDict)
    
    if ifdecomp==False:
        return tDict
        
    '''THIS IS WHERE THE CONVENTION KICKS IN'''
    if defconv=='ad':
        Plist, Qlist, E0=UNIFY_PLIST(a, b, d, c, n, 64*epsifejer)
    elif defconv=='ac':
        Plist, Qlist, E0=UNIFY_PLIST(a, b, c, d, n,64*epsifejer)
    
    tDict['Plist']=Plist
    tDict['Qlist']=Qlist
    tDict['E0']=E0
    
    return Plist, Qlist, E0, a, b, c, d, tDict

def NORM_EXTRACT(a, b, c, d, n, data, epsi):
    """
    Computes the difference between the QSP simulation calculation and the polynomial a+ib at each point
    beginning from coefficient lists, computes the projector set, the QSP solution for each point,
    and the difference between them
    inputs:
    a, b, c d length 2n+1 numpy arrays, should be real-on-circle, pure Laurent polynomials
    n: float, max degree
    data: the number of \theta points used
    epsi: the solution tolerance for computing P, Q, E0

    return: np array with the Euclidean trace distance at each point
    """
    ftestreal=lpf.LAUR_POLY_BUILD(a, n, np.exp(1j*data))
    ftestimag=lpf.LAUR_POLY_BUILD(b, n, np.exp(1j*data))
    
    #Plist, Qlist, E0=BUILD_PLIST(a, b, d, c, n, dofourier=False)
    #Plist, Qlist, E0=UNIFY_PLIST(a, b, c, d, n, epsi/(2*n+1)/16/np.pi)
    Plist, Qlist, E0=BUILD_PLIST(a, b, c, d, n)
    Wlist=Ep_PLOT(Plist, Qlist, E0,n, a, b, data, just_vals=True)
    
    return mf.NORM_CHECK(Wlist, ftestreal+1j*ftestimag)

def NORM_EXTRACT_FROMP(Plist, Qlist, E0,a, b, n,  fcnvals, data):
    """
    Computes the difference between the QSP simulation calculation and the polynomial a+ib at each point
    given known projector sets computes the QSP solution for each point
    
    inputs:
    Plist, Qlist: np 2x2x2n array, must have PList[2, 2, j] be a projector for every j (and for Qlist)
    E0: np 2x2 array, must be in SU(2)
    a, b, length 2n+1 numpy arrays, should be real-on-circle, pure Laurent polynomials
    n: float, max degree
    data: the number of \theta points used

    return: np array with the Euclidean trace distance at each point
    """
    Wlist=Ep_PLOT(Plist, Qlist, E0,n, a, b, data, just_vals=True)
    return  mf.NORM_CHECK(Wlist, fcnvals)

