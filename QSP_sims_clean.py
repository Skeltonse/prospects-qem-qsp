###IMPORTS###
import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms import fold_global
import os
import csv
import matplotlib.pyplot as plt
import pickle
import simulators.matrix_fcns as mf
import functions.laur_poly_fcns as lpf
import parameter_finder as pf
from pathlib import Path
from scipy.optimize import curve_fit

from trotter import generate_hamiltonian

import solvers.Wilson_method as wm
from simulators.projector_calcs import Ep_PLOT, SU2_CHECK, UNIFY_PLIST, BUILD_PLIST
###DEFINE CLASSES FOR HAMILTONIANS AND QSP ORACLES, AND ALSO FOR QSP PARAMETERS###
class ham_oracle_TI_load:
    """
    builts systq-qubit Ising hamiltonian
    then builts $U=exp(1j*arccos(H))$, an oracle for QSP. Uses classical eigendecomp, rather than a heuristic for building QSP oracles
    the class contains:
    systq: number of qubits used to describe the systemHamiltonian
    ancq: number of ancillas used for the block encoding, here 0
    subnorm: the subnormalization of the block encoding

    after DEFINE_ORACLE runs:
    U: the QSP block encoding
    Uevals: its eigenvalues
    evecs: the eigenvectors of H
    Hevals: the eigenvalues of H, after subnormalization
    """
    def __init__(self, systq):
        self.systq = systq
        self.ancq=0
        self.subnorm=1
        
        return

    def DEFINE_ORACLE(self):
        self.H=qml.matrix(generate_hamiltonian(self.systq))
        self.U, self.Uevals, self.Hevals,self.W, self.evecs=mf.UNITARY_BUILD(self.H/self.subnorm, return_evals=True)
        return 

class qsp_params:
    """ 
    class which generates and stores the QSP parameters for a Hamiltonian Simulation problem
    coeffs for P=a+ib generated in Julia as Jacobi-Anger expansion coefficients and then Q computed using 
    Berntson and Suenderhof code
    arguements:
    tau: the simulation time
    approxprecision: the desired precision of the approximation, given as approxprecision where $\epsilon+10**(-approxprecision)$
    device: a keyword arguement to get around mac/window path incompatibilities

    the class contains:
    tau, precision, device: tau, approxprecision, device
    filename: string filename of the coefficients of P
    a, b: coeffiicent lists of Laurent polynomials
    P,Q: coefficient lists of polynomials st $|P|^2+|Q|^2=1$
    n: max degree of the Laurent polynomials
    thetalist, philist, lamba: G-QSP parameters
    """ 
    def __init__(self, tau, approxprecision=4, degree=405, subnorm=0.99999999, lchoice='degree', orderval=5):
        self.tau=tau
        self.subnorm=subnorm
        self.lchoice=lchoice
        if lchoice=='degree':
            self.filename="hsim_coeffs_deg_" +str(degree) + "_t_" + str(tau) 
        elif lchoice=='order':
            self.filename="hsim_coeffs_eorder_" + str(orderval) + "_t_" + str(tau) 
        elif self.precision>=5:
            self.filename="hsim_coeffs_epsi_" + "1.0e-"+str(approxprecision) + "_t_" + str(tau)
        else:
            self.filename="hsim_coeffs_epsi_" +str(10**(-approxprecision)) + "_t_" + str(tau)
        
        pathname="QSP-sims-clean.py"
        inst_tol=10**(-14)
        
        current_path = os.path.abspath(__file__)
        parentdir=os.path.dirname(current_path)
        self.parentdir=parentdir
        datadir=os.path.join(parentdir, "csv_files")
        
        abs_path=os.path.join(datadir, self.filename+".csv")
        
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
        ###GET RID OF ANY INCONVENIANT SMALL COEFFICIENTS###
        a=np.array(column1).astype(float)
        b=np.array(column2).astype(float)
        
        n=int(column3[0])
        self.epsi=column4[0]
        coeffcutoff=10**(-16)
        while abs(a[0])<coeffcutoff and abs(b[0])<coeffcutoff:
            a=a[1:2*n]
            b=b[1:2*n]
            n=n-1
        self.a=subnorm*(2)*a
        self.b=subnorm*(2)*b
        self.n=n
        
        ###getting a QSP circuit
        theta=np.linspace(-np.pi,np.pi, 100)
        cz2list=lpf.LAUR_POLY_MULT(self.a, self.a)
        sz2list=lpf.LAUR_POLY_MULT(self.b, self.b)
        add1=np.append(np.append(np.zeros(2*self.n), 1), np.zeros(2*self.n))
        abunc=cz2list+sz2list
        abun=lpf.LAUR_POLY_BUILD(abunc, 2*self.n,  np.exp(1j*theta))
        self.calFc=add1-abunc
        gammaW, itW, initgamma, nu, Ttilde=wm.WILSON_LOOP_WCHECK(np.real(self.calFc), 2*self.n, nu=inst_tol, init="Wilson", datatype='float')
        self.c, self.d, probflag=pf.cd_PROCESS(gammaW, self.a, self.b,self.n, tol=inst_tol,plots=False)
        self.Plist, self.Qlist, self.E0=UNIFY_PLIST(self.a, -self.b, self.c, self.d, self.n, 64*inst_tol)

        
        
        return

###THINGS WE CAN CHANGE - DEFINE H, MMNT OPERATORS, AND THE PENNYLANE DEVICE###
systq=8
preciexp=4
noisepergate= 0


Hclass=ham_oracle_TI_load(systq)
Hclass.DEFINE_ORACLE()
mmntop=np.kron(np.kron(np.array([[1, 0], [0,1]]), np.array([[1, 0], [0,-1]])), np.kron(np.array([[1, 0], [0,-1]]), np.identity(2**(Hclass.systq-3))))
dev=qml.device("default.mixed", wires=Hclass.systq+Hclass.ancq+1)
dev2=qml.device("default.mixed", wires=Hclass.systq+Hclass.ancq)

###FUNCTIONS THAT SIMULATE AND MEASURE THE QSP CIRCUITS###
def PROMOTE_EVEC(evecs, l):
    """
    promotes an initial state build from an eigenvalue of H to a density matrix.
    evecs: an array with all of the eigenvectors of some matrix
    l: a column index which specifies the  eiegnstate
    """
    return evecs[:, l][:, np.newaxis]@np.conj(evecs[:, l][:, np.newaxis]).T

def CHEAT_QSP(Hclass, QSPpars):
    """ 
    classically computes the desired output of the QSP circuit, ie, builds matrix $D=Qdiag(e^{i\tau \lambda})Q^{\dag}$
    args:
    Hclass: the hamiltonian class
    QSP pars: the QSP parameter class
    returns:
    np array for matrix $D$
    """ 
    dim=2**Hclass.systq
    D=np.zeros([dim, dim], dtype=complex)
    
    newevals= np.exp(-1j*QSPpars.tau*Hclass.Hevals)*QSPpars.subnorm
    return Hclass.evecs@np.diag(newevals)@Hclass.W

def CHEAT_QSP_U(Hclass, QSPpars):
    """ 
    classically computes the desired output of the QSP circuit, ie, builds matrix $D=Qdiag(e^{i\tau \lambda})Q^{\dag}$
    args:
    Hclass: the hamiltonian class
    QSP pars: the QSP parameter class
    returns:
    np array for matrix $D$
    """ 
    dim=2**Hclass.systq
    D=np.zeros([dim, dim], dtype=complex)
    
    newevals= np.exp(-1j*QSPpars.tau*Hclass.Hevals)*QSPpars.subnorm
    return Hclass.evecs@np.diag(newevals)@Hclass.W


@qml.qnode(dev)
def LQSP_PENNY(rho, hclass, qsppars, p2, Ul):
    """ 
    simulates the G-QSP circuit with noise in pennylane, ie, builds an approximation of $\sum_{\lambda}e^{i\tau \lambda} \P_{\lambda}$
    args:
    rho: density matrix for the initial state of the system qubits
    hclass: the hamiltonian class
    qsppars: the QSP parameter class
    pU: probaility of failure for a gate
    returns:
    np array for evolved density matrix $U_{QSP}^{\dag}\rhoU_{QSP}$ 
    """ 
    
    ####NOW WE ACTUALLY RUN THE CIRUIT
    qml.QubitDensityMatrix(rho, wires=range(hclass.systq+1))
    
    for ind in range(int(p2/2)):
        qml.QubitUnitary(np.kron(qsppars.Plist[:, :, p2-2*ind-1], np.identity(Ul))+np.kron(qsppars.Qlist[:, :, p2-2*ind-1],np.conj(qml.matrix(hclass.Up)).T), wires=range(hclass.systq+1))
        qml.QubitUnitary(np.kron(qsppars.Qlist[:, :, p2-2*ind-2], np.identity(Ul))+np.kron(qsppars.Plist[:, :, p2-2*ind-2],qml.matrix(hclass.Up)), wires=range(hclass.systq+1))
        

    qml.QubitUnitary(qsppars.E0, wires=[0])
    return qml.density_matrix(wires=range(hclass.systq+1))#qml.expval(qml.X(1)), qml.var(qml.X(1))#qml.exp(mmntop)#np.trace(mmntop@rhored)

# note that number of qubits has to be manually changed
def MEASURE_NOISY_QSP(rho, mmnt, shots_actual):
    dev_in = qml.device("default.mixed", wires=8, shots=shots_actual)  
    @qml.qnode(dev_in)
    def MEASURE_NOISY_QSP_in():
        qml.QubitDensityMatrix(rho, wires=range(8))
        return qml.expval(mmnt)
    return MEASURE_NOISY_QSP_in()

@qml.qnode(dev2)
def EXACT_MEASURE_NOISY_QSP(rho,hclass, mmnt):
    """
    measure the observable using inbuilt pennylane functions
    the variance doesn't work without using inbuilt pennylane functions, but I think this is fine as long as we're careful with definitions.
    that is,  observable=qml.Hermitian(mmnt, range(hclass.systq)) works in qml.expval but not in qml.var

    input: the reduced density matrix over qubits corresponding to the Hamiltonian state space
    hclass: class with information on the Hamiltonian
    mmnt: a pennylane operator, the measurement we want

    returns:
    the expectation value Tr(mmnt*rho) and its variance 
    """
    #observable=qml.Hermitian(mmnt, range(hclass.systq))
    qml.QubitDensityMatrix(rho, wires=range(hclass.systq))
    return qml.expval(mmnt)

# define extrapolation models for the mitigation
def exponential_model(x, a, b):
    return a * np.exp(-1* b * x)

def linear_model(x, a, b):
    return b*x+a

def richardson(y, c):
    mitigated = 0
    parameters = []
    for k in range(0, len(y)):
        product = 1
        for i in range(0, len(y)):
            if k != i:
                product = product * (c[i]/(c[k]-c[i]))
        parameters.append(product)
        mitigated = mitigated + y[k]*product
    return mitigated, parameters


# run mitigation 
# the number of qubits needs to be changed manually
# generate scaled circuits, add noise to folded circuits and extract expectation values
def RUN_NOISY_PENNY(rho, hclass, qsppars, pU, mmnt,  scaling, p2, Ul):
    # Get the tape from the QNode
    LQSP_PENNY(rho, hclass, qsppars, p2, Ul)
    tape = LQSP_PENNY.qtape  
    depth = tape.graph.get_depth()  
    print(depth)

    # calculate number of shots
    M= 5e6
    number_shots = int(M)
    print(number_shots)

    # collect data for mitigation
    expdict={}
    var_y = []
    n_or = 1
    for c in scaling:
        #folded = qml.transforms.fold_global(QSP_PENNY, c)

        # build noise model for different noise c*p
        fcond0 = qml.noise.wires_in(range(hclass.systq+1))
        noise0 = qml.noise.partial_wires(qml.DepolarizingChannel, c*pU)
        noise_model = qml.NoiseModel({fcond0: noise0})

        noisy_circuit = qml.add_noise(LQSP_PENNY, noise_model)
        rho_for_meas = noisy_circuit(rho, hclass, qsppars, p2, Ul)
    
        projtoconvent=np.conj(np.kron(qsppars.qspancmatrix, np.identity(2**hclass.systq))).T@rho_for_meas@np.kron(qsppars.qspancmatrix, np.identity(2**hclass.systq))
        probofoutcome=np.trace(projtoconvent)
        rhored=np.trace((projtoconvent/probofoutcome).reshape(2,2**hclass.systq, 2, 2**hclass.systq), axis1=0, axis2=2)
        print(np.shape(rhored))
        print(np.shape(mmnt))
        e = MEASURE_NOISY_QSP(rhored,qml.Hermitian(mmnt, range(Hclass.systq)), number_shots)
        var_e = np.trace(mmnt@mmnt@rhored)-(np.trace(mmnt@rhored))**2
        print(var_e)

        # saving e FOR C
        expdict["noisyexpectationarray_"+str(c)]=e

        var_y.append(var_e)
    var_final = var_y
    expdict['variance']=var_final

    return expdict

###THE FUNCTION WHICH RUNS AND MEASURES THE QSP CIRCUIT###
def RUN_QSP(Hclass, QSPpars, mmntop, pU, scaling, numstates=0, ifsave=False, Hlabel="", pennyop=False, ifplots=False):
    """ 
    Function which generates and plots all of the expectation values
    args:
    hclass: the hamiltonian class
    qsppars: the QSP parameter class
    mmntop: the measurement operator as np array with the same dimensions as $H$
    mmntpenny: the measurement operator as a pennylane Pauli term over qubits it acts upon
    pU: probaility of failure for a gate
    numstates: determines the number of different inital points to sample. numpoints=0 is the default and 
    uses every eiegnvector as an initial point

    returns:

    """ 
    ###DEFINE ARRAYS TO STORE EXPECTATION VALUES AND A COUNTER FOR THE NUMBER OF DIFFERENT STATES TO GENERATE###
    if numstates==0:
        counter=len(Hclass.evecs)
    else: 
        counter=numstates

    cheatvals=np.zeros(counter, dtype=complex)
    qspvals=np.zeros(counter, dtype=complex)
    qspvars1=np.zeros(counter, dtype=complex)

    ####WHY DO BOTH OF THESE WORK???###
    qspancmatrix=np.array([[1, 1], [1, 1]])/2
    # qspancmatrix=np.array([[1, 0], [0, 0]])

    QSPpars.qspancmatrix=qspancmatrix
    
    p0,p1,p2=QSPpars.Plist.shape
    
    ###COMPUTE NOISELESS UNITARY### 
    if pennyop==False:
        D=(1/QSPpars.subnorm)*CHEAT_QSP(Hclass, QSPpars)
        Hclass.Up=qml.QubitUnitary(Hclass.U, wires=range(1, Hclass.systq+1))
        Hclass.Udag=qml.QubitUnitary(np.conj(Hclass.U).T, wires=range(1, Hclass.systq+1))
    elif pennyop==True:
        D=(1/QSPpars.subnorm)*CHEAT_QSP_U(Hclass, QSPpars)
        Hclass.Up=Hclass.U
        Hclass.Udag=qml.adjoint(Hclass.Up.adjoint())

    ####RUN QSP SIMULATIONS AND COMPUTE EXPECTATION VALUES
    savedict={'Hdict': Hclass.__dict__, 'QSPdict':QSPpars.__dict__, 'noisepergate': pU, 'measuredobservable':mmntop, 'scaling': scaling, 'numinitialstates':numstates}
    for l in range(0, counter):
        if numstates==0:
            initsystrho=PROMOTE_EVEC(Hclass.evecs, l)
           
        else:
            numterms=len(Hclass.Hevals)
            initsystrho=np.zeros([len(Hclass.Hevals), len(Hclass.Hevals)], dtype=complex)
            initsystrho[l, l]=1
        
        savedict['rhoinit_'+str(l)]=initsystrho

        ###COMPUTE THE NOISELESS EXPECTATION VALUES
        rhotest= D@initsystrho@np.conj(D).T
        cheatvals[l]=(np.trace(mmntop@rhotest))

        # save dict here and read off c=1 for noisy value and call mitigation function for others
        #qspvals[l], qspvars[l]=MEASURE_NOISY_QSP(rhored, Hclass,   mmntpenny)
        initrho=np.kron(qspancmatrix, initsystrho)
        
        k=QSPpars.n
        Ul=len(Hclass.H)
        savedict['noisyexpdict_'+str(l)]=RUN_NOISY_PENNY(initrho, Hclass, QSPpars, pU, mmntop,  scaling, p2, Ul)
        savedict['noiselessexpdict']=qspvals
        
        # ###COMPUTE THE NOISY EXPECTATION VALUES###
        
        qsprho=LQSP_PENNY(initrho, Hclass, QSPpars, p2, Ul)
        probofoutcome=np.trace(np.kron(QSPpars.qspancmatrix, np.identity(2**Hclass.systq))@qsprho)
        rhored=np.trace((np.kron(QSPpars.qspancmatrix, np.identity(2**Hclass.systq))@qsprho/probofoutcome).reshape(2,2**Hclass.systq, 2, 2**Hclass.systq), axis1=0, axis2=2)
        qspvals[l]=EXACT_MEASURE_NOISY_QSP(rhored, Hclass,  qml.Hermitian(mmntop, range(Hclass.systq)))
        
    savedict['classicalvals']=cheatvals
    if ifsave==True:
        savename=QSPpars.filename.replace("hsim_coeffs", "noisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*pU)))
        savepath=os.path.join(QSPpars.parentdir, "benchmark_data")
        with open(os.path.join(savepath,savename+".pkl"), 'wb') as f:
            pickle.dump(savedict, f)
    
    ###PLOTS###
    if ifplots==True:
    
        if numstates==0:
            indices=np.argsort(Hclass.Hevals)
            xvals=Hclass.Hevals[indices]
            plt.xlabel(r"$\lambda$ of Hamiltonian $H$")

        ##if not, just plot according to increasing value (purely asthetic choice) and set the xaxis to a simple numeric label
        else:
            xvals=range(numstates)
            indices=np.argsort(cheatvals)
            plt.xlabel(r"label $i$ for $\rho_i$")
    
        cheatvals=cheatvals[indices]
        qspvals=qspvals[indices]
        print("comp", qspvals, cheatvals)
        print("diff", abs(qspvals-cheatvals) )
        print(qspvals)
        print(cheatvals)
        plt.plot(xvals, np.real(cheatvals))
        plt.scatter(xvals, np.real(qspvals), color='red')
        plt.title("expectation value vs eigenvalue")
        plt.ylabel("Pauli mmnt on 1st qubit")
        
        plt.show()

        print("difference between actual and noise=0 circuit values", max(abs(qspvals-cheatvals)))
        print("class values,", cheatvals )
    return


taulist = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0]
noiselist = [1e-4, 1e-3, 1e-2]
for noisepergate in noiselist:
    for tau in taulist:
        QSPpars=qsp_params(tau,subnorm=1/np.sqrt(2), lchoice='order', orderval=3)
        RUN_QSP(Hclass, QSPpars, mmntop,noisepergate,[1, 1.1, 1.25, 1.3, 1.5, 1.75,  2, 3], numstates=1, ifsave=True, pennyop=False, Hlabel="TI8_fixed_e2")


def MITIGATION(scaling, l, datadict,  Hlabel="TI4"):
    values = [1, 1.1, 1.25, 1.3, 1.5, 1.75,  2, 3]
    indices = [values.index(value) for value in scaling]
    variance = datadict['noisyexpdict_'+str(l)]["variance"]
    variance = [variance[i]/5e6 for i in indices]
 
    
    y = []
    for c in scaling:
        expectation_noisy = datadict['noisyexpdict_'+str(l)]["noisyexpectationarray_"+str(c)]
        y.append(expectation_noisy)
    

    try: 
        params, _ = curve_fit(exponential_model, scaling, y, maxfev=1000)
        new_x = 0 
        mitigated = exponential_model(new_x, *params) 
        var = params[1]*np.exp(-1*params[1]*mitigated)**2*sum(variance) 
        if np.abs(mitigated)>=1.2:
            print('nonsence in exp, trying richardson')
            mitigated, params = richardson(y, scaling)
            var = variance[0]*params[0]**2+variance[1]*params[1]**2
            if np.abs(mitigated)>=1.2:
                print('Use linear fit, as mitigated is very large')
                params, _ = curve_fit(linear_model, scaling, y)
                new_x = 0 
                mitigated = linear_model(new_x, *params)
                var = sum(variance)*params[1]**2
                #print(mitigated, var)
                if np.abs(mitigated)>=1.2:
                    print('Mitigated value is still out of control, taking the point out')
                    mitigated = 0
                    var = 0

    except RuntimeError as e:
        error_message = str(e)  # Convert exception to string
    
        if "maxfev" in error_message.lower():
            print("Detected maxfev error. Retrying with higher maxfev...")
            try:
                params, _ = curve_fit(exponential_model, scaling, y, maxfev=10000)
                print("Exponential model fit successful with increased maxfev:")
                new_x = 0 
                mitigated = exponential_model(new_x, *params)  
                var = params[1]*np.exp(-1*params[1]*mitigated)**2*sum(variance)

                if np.abs(mitigated)>=1.2:
                    print('nonsence in exp, trying Richardson')
                    mitigated, params = richardson(y, scaling)
                    var = variance[0]*params[0]**2+variance[1]*params[1]**2
                    if np.abs(mitigated)>=1.2:
                        print('Use linear fit, as mitigated is very large')
                        params, _ = curve_fit(linear_model, scaling, y)
                        new_x = 0 
                        mitigated = linear_model(new_x, *params)
                        var = sum(variance)*params[1]**2
                        if np.abs(mitigated)>1.2:
                            print('Mitigated value is still out of control, taking the point out')
                            mitigated = 0
                            var = 0

                if np.abs(mitigated)>=1.2:
                    print('Use linear fit, as mitigated is very large')
                    params, _ = curve_fit(linear_model, scaling, y)
                    new_x = 0 
                    mitigated = linear_model(new_x, *params)
                    var = sum(variance)*params[1]**2
                    print(mitigated, var)
                    if np.abs(mitigated)>=1.2:
                        print('Mitigated value is still out of control, taking the point out')
                        mitigated = 0
                        var = 0
            finally:
                pass 

        else:
            print(" Exponential failed, for different reason than maxfev. Trying Richardson model...")
            mitigated, params = richardson(y, scaling)
            var = variance[0]*params[0]**2+variance[1]*params[1]**2

            print('points', y)
            print('params', params)
            if np.abs(mitigated)>=1.2:
                print('Use linear fit, as mitigated is very large')
                params, _ = curve_fit(linear_model, scaling, y)
                new_x = 0 
                mitigated = linear_model(new_x, *params)
                var = sum(variance)*params[1]**2
                if np.abs(mitigated)>=1.2:
                    print('Mitigated value is still out of control, taking the point out')
                    mitigated = 0
                    var = 0


    #print(mitigated, var)

    return mitigated, var

# function to filter out 0 values of the mitigated (i set it to zero if mitigation fails tremendously)
# make sure first one inpouted ios actually mitigated values in arrays
# then all other values related to it, like variance, taulist and so on
def filter_arrays_by_first_in_place(*arrays):
    first = arrays[0]
    indices_to_keep = [i for i in range(len(first)) if first[i] != 0]

    for arr in arrays:
        # Keep only elements at the positions where first array is not 0
        filtered = [arr[i] for i in indices_to_keep]
        arr[:] = filtered 

        