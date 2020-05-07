import sympy as sp
import numpy as np

def newton_raphson(f, v, x):
    mt = sp.symbols('0')  # The symbolic equivalent of 0
    it_max = 1000  # Maximum number of iterations
    tol = 1e-14  # Tolerance

    if (type(f) == list):
        nofe = len(f)  # Number of equations
        X = np.zeros((it_max, nofe, 1))  # Empty matrix of initial values
        F = np.full((nofe, 1), mt)  # Empty vertical matrix of equations
        V = np.full((nofe), mt)  # Empty horizontal matrix of variables
        for i in range(0, nofe):
            X[0, i, 0] = x[i]  # Filling those three matrices with the input
            F[i, 0] = f[i]
            V[i] = v[i]
    else:
        nofe = 1
        X = np.zeros((it_max, nofe, 1))  # Empty matrix of initial values
        F = np.full((nofe, 1), mt)  # Empty vertical matrix of equations
        V = np.full((nofe), mt)  # Empty horizontal matrix of variables
        for i in range(0, nofe):
            X[0, i, 0] = x  # Filling those three matrices with the input
            F[i, 0] = f
            V[i] = v

    DF = np.full((nofe,nofe), mt)  # An empty square matrix for the Jacobian
    DFX = np.zeros((it_max,nofe,nofe))  # Jacobians with the initial values
    DFX_inv = np.zeros((it_max,nofe,nofe))  # The inverse of that Jacobian
    FX = np.zeros((it_max,nofe,1))  # The input functions of the initial values
    H = np.zeros((it_max,nofe,1))   # A matrix with the residuals

    stop = False    # A boolean variable to end the entire iteration process

    # This loop obtains the derivatives of F
    for i in range(0,nofe):
        for j in range(0,nofe):
            DF[i,j] = sp.diff(F[i,0], V[j])

    # This is where each iteration begins
    for it in range(0,it_max):

        # We start each iteration if all the previous variables don't converge
        if (stop == False):
            conv = 0  # This is how many variables have converged so far
            subby = '{'  # Input for the substitutions done in DFX and FX

            for i in range(0, nofe):
                subby = subby + 'V[' + str(i) + ']: X[it,' + str(i) + ',0]'
                if (i != nofe-1):
                    subby = subby + ',  '
                if (i == nofe-1):
                    subby = subby + '} '

            # Jacobian matrix with the initial values
            for i in range(0,nofe):
                for j in range(0,nofe):
                    DFX[it,i,j] = DF[i,j].subs(eval(subby)).evalf()

            # The input functions with the initial values
            for i in range(0,nofe):
                FX[it,i,0] = F[i,0].subs(eval(subby)).evalf()

            DFX_inv[it] = np.linalg.inv(DFX[it])  # The inverse
            H[it] = np.dot(DFX_inv[it], -FX[it])  # The residuals

            if(it+1 < it_max):
                for i in range(0,nofe):
                    X[it + 1, i, 0] = X[it, i, 0] + H[it, i, 0]  # Update
            else:
                print("It won't converge")
                print("Try with other initial values")
                stop = True  # Stops the loop

            # If each residual is lower than the tolerance, we consider that
            # another variable has reached convergence
            for i in range(0, nofe):
                if(abs(H[it, i, 0]) <= tol):
                    conv = conv + 1

            # If all of the variables have converged we return the final values
            # and stop the loop
            if(conv == nofe):
                if(nofe == 1):
                    return X[it,0,0]
                else:
                    return (X[it,i,0] for i in range(0, nofe))
                stop = True

x, y, z = sp.symbols('x y z')
f1 = 2 * x + 3 * y - 2 * z - 15
f2 = 3 * x - 6 * y - 12 * z - 21
f3 = 7 * x - 3 * y + 19 * z + 30

x, y, z = newton_raphson([f1, f2, f3], [x, y, z], [1, 1, 1])
print(x)
print(y)
print(z)