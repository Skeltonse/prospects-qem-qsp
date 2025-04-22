"""
Useful functions on matrix valued objects
"""
import numpy as np
from scipy import linalg


sigmaX=np.array([[0,1],[1, 0]])
sigmaZ=np.array([[1,0],[0, -1]])
sigmaY=np.array([[0,-1j],[1j, 0]])
I=np.identity(2)

def SENSIBLE_MATRIX(A, tol=10**(-16)):
    """
    function to make reading matrix results easier, sets any very small matrix elements to zero

    inputs:
    A: n x n complex np. array
    tol: tolerance of the solution
    """
    Ar=np.where(abs(np.real(A))>tol, np.real(A), 0)
    Ai=np.where(abs(np.imag(A))>tol, np.imag(A), 0)
    return Ar+Ai*1j

def NORM_CHECK(qeta, fnca):
    """
    computes the distance between points in C^2 and returns the maximum

    inputs:
    qeta, fnca: np arrays, assumed to be lists of functional values at different points in the complex plane

    return: the maximum distance between the pointwise functional values
    """
    obj=qeta-fnca
    normlist=np.sqrt(np.real(obj)**2+np.imag(obj)**2)
    return np.max(normlist)

def UNITARY_BUILD(H, return_evals=False, check_decomp=False):
    """
    Builds e^{iarccos(H)}, a suitable oracle for complex QSP

    input:
    H: Hermitian matrix
    return_evals: option to return the eigenvalues of H and the eigenvectors

    return:
    U: np array, the unitary QSP oracle for H
    Uevals: np array with the eigenvalues of U
    """
    dims=np.shape(H)[0]
    # Hevals, evecs=np.linalg.eig(H)
    Hevals, evecs=linalg.eig(H, right=True) #scipy one

    print("is eigdecomp working", np.allclose(evecs@np.diag(Hevals)@np.conj(evecs).T, H, 10**(-14))) #this is fine

    Uevals=np.exp(1j*np.arccos(Hevals))
    U=evecs@np.diag(Uevals)@np.conj(evecs).T
    
    if check_decomp==True:
        D=np.zeros([16, 16], dtype=complex)
        for l in range(0, len(H)):
            evecproj= evecs[:, l][:, np.newaxis]@np.conj(evecs[:, l][:, np.newaxis]).T
            print("is this evec ("+str(l)+") normalized?")

            Hlvec=evecs[:, l][:, np.newaxis]
            for j in range(0, len(H)):
                print(SENSIBLE_MATRIX(np.conj(Hlvec).T@evecs[:, j][:, np.newaxis], 10**(-14)))
            D=D+evecproj
        print("is D identity", np.where((SENSIBLE_MATRIX(D, 10**(-14))!=0) & (abs(SENSIBLE_MATRIX(D, 10**(-14))-1)>10**(-12))))

    if return_evals==True:
        return U, Uevals, Hevals, np.conj(evecs,).T, evecs
    
    return U, Uevals

