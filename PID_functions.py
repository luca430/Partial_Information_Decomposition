import sys
import scipy.io
import numpy as np
from numpy.linalg import det

#Mutual information between target X and source Y
def I_XY(X,Y):
    
    dim_X = X.ndim
    dim_Y = Y.ndim
    
    cov = np.cov(X,Y)
    if (dim_X == 1):
        det_sigma_X = cov[0,0]
    elif (dim_X > 1):
        det_sigma_X = np.linalg.det(cov[:dim_X,:dim_X])
    if (dim_Y == 1):
        det_sigma_Y = cov[dim_X,dim_X]
    elif (dim_Y > 1):
        det_sigma_Y = np.linalg.det(cov[dim_X:,dim_X:])
        
    I = np.log2((det_sigma_X*det_sigma_Y)/np.linalg.det(cov))
    
    return I

#Partial Information Decomposition as described by Barrett
def PID(X,Y,Z):
    I_xy = I_XY(X,Y)
    I_xz = I_XY(X,Z)
    YZ = np.vstack((Y,Z))
    I_xyz = I_XY(X,YZ)
    
    R = min(I_xy,I_xz)
    
    if R == I_xy:
        Uy = 0
        Uz = I_xz - I_xy
        S = I_xyz - I_xz   
    else:
        Uz = 0
        Uy = I_xy - I_xz
        S = I_xyz - I_xy
    return Uy, Uz, S, R

#When analysing multiple time series, creates 3 matrices for the components of the PID for every couple (i,j) of TS.
# s_target specifies a single target for the PID as the i-th time series in the loop.
# d specifies the time delay between source and target.
def SRU(X,s_target=True,d=1):    
    X_prev = X[:,:-d]
    X_succ = X[:,d:]
    S_mat = np.zeros((len(X),len(X)))
    R_mat = np.zeros((len(X),len(X)))
    U_mat = np.zeros((len(X),len(X)))

    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                if (s_target):
                    Uy, Uz, S, R = PID(X_succ[i,:],X_prev[i,:],X_prev[j,:])
                else:
                    X_12 = np.vstack((X_succ[i,:],X_succ[j,:]))
                    Uy, Uz, S, R = PID(X_12,X_prev[i,:],X_prev[j,:])
                S_mat[i,j] = S
                R_mat[i,j] = R
                if (Uy > Uz):
                    U_mat[i,j] = Uy
                else:
                    U_mat[i,j] = -Uz
    return(S_mat, R_mat, U_mat)

#Transfer entropy computed from the Unique and Synergetic contributions of PID
def transfer_entropy(U_mat, S_mat):
    T_YX = np.zeros((len(S_mat),len(S_mat)))
    for i in range(len(S_mat)):
        for j in range(len(S_mat)):
            if U_mat[i,j] < 0:
                T_YX[i,j] = -U_mat[i,j] + S_mat[i,j]
            else:
                T_YX[i,j] = S_mat[i,j]
    return(T_YX)

#PID with different time intervals
def SRU_delay(X,d):    
    X_prev = X[:,:-d]
    X_succ = X[:,d:]
    S_mat = np.zeros((len(X),len(X)))
    R_mat = np.zeros((len(X),len(X)))
    U_mat = np.zeros((len(X),len(X)))

    for i in range(len(X)):
        for j in range(i):
            if i != j:
                X_12 = np.vstack((X_succ[i,:],X_succ[j,:]))
                Uy, Uz, S, R = PID(X_12,X_prev[i,:],X_prev[j,:])
                S_mat[i,j] = S
                R_mat[i,j] = R
                if (Uy > Uz):
                    U_mat[i,j] = Uy
                else:
                    U_mat[i,j] = -Uz
                    
    S_mat += S_mat.T
    R_mat += R_mat.T
    U_mat -= U_mat.T
    
    return(S_mat, R_mat, U_mat)

# PHID decomposition
# the output is a dictionary containing all the atoms
def PHID(X1,X2,Y1,Y2):
    # X and Y have dim (2,m) where X represents the past and Y is the future
    
    X = np.vstack((X1,X2))
    Y = np.vstack((Y1,Y2))

    I1_1 = I_XY(X1,Y1)
    I1_2 = I_XY(X1,Y2)
    I2_1 = I_XY(X2,Y1)
    I2_2 = I_XY(X2,Y2)
    I1_12 = I_XY(X1,Y)
    I2_12 = I_XY(X2,Y)
    I12_1 = I_XY(X,Y1)
    I12_2 = I_XY(X,Y2)
    I12_12 = I_XY(X,Y)

    R_R = min([I1_1,I1_2,I2_1,I2_2])

    u1, u2, s, R12_1 = PID(Y1,X1,X2)  #notation: R(past)_(future)
    u1, u2, s, R12_2 = PID(Y2,X1,X2)
    u1, u2, s, R12_1c2 = PID(Y,X1,X2) # 1c2 means 1 joint with 2 in the target aka Y instead of Y1 or Y2
    u1, u2, s, R1_12 = PID(X1,Y1,Y2)
    u1, u2, s, R2_12 = PID(X2,Y1,Y2)
    u1, u2, s, R1c2_12 = PID(X,Y1,Y2)

    A = np.array([[1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                  [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                  [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
                  [1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0],
                  [1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0],
                  [1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0],
                  [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
                  [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                  [1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1],
                  [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
                  [1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1],
                  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

    B = np.reshape([R12_1,R12_2,R12_1c2,R1_12,R2_12,R1c2_12,
                   I1_1,I1_2,I2_1,I2_2,I12_1,I12_2,I1_12,I2_12,I12_12,
                   R_R], (16,1))

    sol = np.dot(np.linalg.inv(A),B)
    sol = np.reshape(sol,(16,))

    variables = ['R_R','R_S','R_U1','R_U2','S_R','S_S','S_U1','S_U2',
                'U1_R','U1_U1','U1_U2','U1_S','U2_R','U2_S','U2_U1','U2_U2']
    dictionary = dict(zip(variables,sol))

    return dictionary

# PHID decomposition for multiple time series. The output is a dictionary containing all the matrices representing the atoms.
def PHID_m(X):
    
    P = X[:,:-1]
    F = X[:,1:]
    
    R_R = np.zeros((len(X),len(X)))
    R_S = np.zeros((len(X),len(X)))
    R_U1 = np.zeros((len(X),len(X)))
    R_U2 = np.zeros((len(X),len(X)))
    S_R = np.zeros((len(X),len(X)))
    S_S = np.zeros((len(X),len(X)))
    S_U1 = np.zeros((len(X),len(X)))
    S_U2 = np.zeros((len(X),len(X)))
    U1_R = np.zeros((len(X),len(X)))
    U1_S = np.zeros((len(X),len(X)))
    U1_U1 = np.zeros((len(X),len(X)))
    U1_U2 = np.zeros((len(X),len(X)))
    U2_R = np.zeros((len(X),len(X)))
    U2_S = np.zeros((len(X),len(X)))
    U2_U1 = np.zeros((len(X),len(X)))
    U2_U2 = np.zeros((len(X),len(X)))
    
    matrices = [R_R,R_S,R_U1,R_U2,
                S_R,S_S,S_U1,S_U2,
                U1_R,U1_S,U1_U1,U1_U2,
                U2_R,U2_S,U2_U1,U2_U2]
    
    variables = ['R_R','R_S','R_U1','R_U2','S_R','S_S','S_U1','S_U2',
                'U1_R','U1_U1','U1_U2','U1_S','U2_R','U2_S','U2_U1','U2_U2']
    dictionary = dict(zip(variables,matrices))
    
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                phid = PHID(P[i,:],P[j,:],F[i,:],F[j,:])
                for key in dictionary.keys():
                    dictionary[key][i,j] = phid[key]
            
    return dictionary

# function that create the frames used for the gif
def create_frame(t,M,name,title=None):
    fig, ax = plt.subplots(figsize=(8,8))
    if name == "u":
        cax = ax.matshow(M, cmap='RdBu', vmin = -2, vmax = 2)
    else:
        cax = ax.matshow(M, vmin = 0, vmax = 2)
    fig.colorbar(cax)
    ax.set_title(f"{title} t = {t}")
    plt.savefig(f'img_{t}_{name}.png', 
                transparent = False,  
                facecolor = 'white'
               )
    plt.close()