import pennylane as qml
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os
import pickle

from functools import lru_cache

dt = 0.1
N = 8

def generate_hamiltonian(N, J_xx=(0.1*-0.2*7/10), J_zz=-0.2*7/10, h=(0.1*-0.2*7/10)):

    coeffs = []
    ops = []
    
    # Nearest-neighbor XX and ZZ interactions
    for i in range(N - 1):  
        coeffs.append(J_xx)
        ops.append(qml.PauliX(i) @ qml.PauliX(i + 1))

        coeffs.append(J_zz)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    # On-site X terms
    for i in range(N):
        coeffs.append(h)
        ops.append(qml.PauliX(i)) 
    return qml.Hamiltonian(coeffs, ops)

# Example usage
H = generate_hamiltonian(N)
# turn H into matrix 
H_matrix = qml.matrix(H)


# Apply Trotterized evolution

dev = qml.device("default.mixed", wires=N, shots=1000)
dev2 = qml.device("default.mixed", wires=N)

def apply_depolarizing_noise(prob, wires):
    for wire in wires:
        qml.DepolarizingChannel(prob, wires=wire)


@qml.qnode(dev2)
@qml.simplify
def trotter_circuit_zzxz_noisy(N,  time, dt, Jx, Jz, hx,  noise_prob):
    qml.BasisState(np.zeros(N, dtype=int), wires=range(N))
    steps = round(time/dt)
    if steps==0:
        return
    else: 
        for i in range(steps):
            for l in range(0, N, 2):
                qml.IsingZZ(-2*dt*Jz, [l, l+1])
                apply_depolarizing_noise(noise_prob, wires=[l, l+1])
            for m in range(1, N-1, 2):
                qml.IsingZZ(-2*dt*Jz, [m, m+1])
                apply_depolarizing_noise(noise_prob, wires=[m, m+1])
            for l in range(0, N, 2):
                qml.IsingXX(-2*dt*Jx, [l, l+1])
                apply_depolarizing_noise(noise_prob, wires=[l, l+1])
            for m in range(1, N-1, 2):
                qml.IsingXX(-2*dt*Jx, [m, m+1])
                apply_depolarizing_noise(noise_prob, wires=[m, m+1])
            for n in range(0, N):
                qml.RX(-2*dt*hx, n)
                apply_depolarizing_noise(noise_prob, wires=[n])

    return qml.density_matrix(range(N))


def MEASURE_NOISY_QSP(rho, mmnt, shots_actual):
    dev_in = qml.device("default.mixed", wires=N, shots=shots_actual)  
    @qml.qnode(dev_in)
    def MEASURE_NOISY_QSP_in():
        qml.QubitDensityMatrix(rho, wires=range(N))
        return qml.expval(mmnt)
    return MEASURE_NOISY_QSP_in()

@qml.qnode(dev2)
def EXACT_MEASURE_NOISY_QSP(rho,mmnt):
    qml.QubitDensityMatrix(rho, wires=range(N))
    return qml.expval(mmnt), qml.var(mmnt)

def RUN_NOISY_TROTTER(H, Jx, Jz, hx, N, time, dt, p, scaling, mmnt):
    # define dict for saving
    expdict={}

    # depth for calculation of the shpot calculations (need ideal circuit)
    # Run the circuit once to create the tape
    trotter_circuit_zzxz_noisy(N,  time, dt, Jx, Jz, hx,  0)

    # Get the tape from the QNode
    tape = trotter_circuit_zzxz_noisy.qtape  # Extract the tape
    depth = tape.graph.get_depth()  # Compute circuit depth
    print(depth)

    # calculate number of shots
    '''
    delta = dt**2
    epsilon = 3*delta
    Mup = np.log10(2/epsilon)
    lower= (1-(p*scaling[-1]))**(depth*2)
    Mdown = lower*4*np.log(2)*N*3**2*delta**2
    M = Mup/Mdown
    '''
    #fixed number of shots
    M = 5e6
    number_shots = int(M)


    # generate ideal trotter circuit and get values
    ideal_rho = trotter_circuit_zzxz_noisy(N,  time, dt, Jx, Jz, hx,  0)
    mmnt_matrix = qml.matrix(mmnt)
    E_ideal, var_ideal = EXACT_MEASURE_NOISY_QSP(ideal_rho, mmnt)
    var_ideal = np.trace(mmnt_matrix@mmnt_matrix@ideal_rho)-(np.trace(mmnt_matrix@ideal_rho))**2
    expdict["idealexpectationvalue"]=E_ideal
    expdict["idealvariance"]=var_ideal


    # collect data for mitigation
    var_y = []
    n_or = 1
    for c in scaling:
        # build noise model for different noise c*p
        rho_for_meas = trotter_circuit_zzxz_noisy(N,  time, dt, Jx, Jz, hx,  c*p)
        var_e = np.trace(mmnt_matrix@mmnt_matrix@rho_for_meas)-(np.trace(mmnt_matrix@rho_for_meas))**2
        # perform measurment with correct number of shots
        e = MEASURE_NOISY_QSP(rho_for_meas, mmnt, number_shots)
        # saving e FOR C
        expdict["noisyexpectationarray_"+str(c)]=e
        var_y.append(var_e)

    var_final = var_y
    expdict['variance']=var_final
    expdict['depth']=depth

    #expdict['shots']=number_shots
    
    return expdict

# run and save full results
def run_full_trotter(N, H, Jx, Jz, hx, mmnt, dt,  time, p, scaling, lchoice='degree', Hlabel="TI4"):
    # create dict to save in
    savedict={ 'noisepergate': p, 'measuredobservable':qml.matrix(mmnt), 'scaling': scaling, 'dt': dt, 'time': time, 'N': N, 'Hamiltonian': qml.matrix(H)}
    savedict['noisyexpdict_'+str(time)] = RUN_NOISY_TROTTER(H, Jx, Jz, hx, N, time, dt, p, scaling, mmnt)

    approxprecision=2
    if lchoice=='degree':
        filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*p))+"_deg_" +str(5) + "_t_" + str(tau) 
    elif lchoice=='order':
        filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*p))+"eorder_" + str(5) + "_t_" + str(tau) 
    elif self.precision>=5:
        filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*p))+"_epsi_" + "1.0e-"+str(approxprecision) + "_t_" + str(tau)
    else:
        filename="trotternoisymmnts_shots_"+Hlabel+"p="+str(np.int64(10000*p))+"_epsi_" +str(10**(-approxprecision)) + "_t_" + str(tau)
    
    
    current_path = os.path.abspath(__file__)
    parentdir=os.path.dirname(current_path)    
    savepath=os.path.join(parentdir, "benchmark_data")
    with open(os.path.join(savepath,filename+".pkl"), 'wb') as f:
        pickle.dump(savedict, f)
    del savedict
    return 

# choose correct parameters
#taulist = np.arange(0.1, 5, 0.1)
taulist = np.arange(5.2, 20, 0.2)
noiselist = [1e-4, 1e-3, 1e-2]


# check that taus in taulist are divisible by dt!!!!
# for noisepergate in noiselist:
#     for tau in taulist:

#         run_full_trotter(N, H, (0.1*-0.2*7/10), -0.2*7/10, (0.1*-0.2*7/10), (qml.Identity(0)@qml.PauliZ(1)@qml.PauliZ(2)@qml.Identity(3)@qml.Identity(4)@qml.Identity(5)@qml.Identity(6)@qml.Identity(7)), dt,  tau, noisepergate, [1, 1.1, 1.25, 1.3, 1.5, 1.75,  2, 3], lchoice='degree', Hlabel="TI8_dt01_fixed")

