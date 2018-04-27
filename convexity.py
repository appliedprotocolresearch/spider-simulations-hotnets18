import autograd.numpy as np
from autograd import grad
from autograd import hessian_vector_product
from scipy.optimize import minimize
from autograd import elementwise_grad as egrad

def reset_rate(x, y, c):
	return (x-y)/((1 - (y/x)**(c/2.0))*c/(1 - (y/x)**c) - (c/2.0))

def determinant(x, y, c, dr2_dx2, dr2_dy2, dr2_dxdy):
        a = dr2_dx2(x, y, c)
        b = dr2_dy2(x, y, c)
        d = dr2_dxdy(x, y, c)

        return a*b - d*d

def main():
        # compute all the gradients
        dr_dx = grad(reset_rate, 0)
        dr_dy = grad(reset_rate, 1)
        dr2_dx2 = grad(dr_dx, 0)
        dr2_dy2 = grad(dr_dy, 1)
        dr2_dxdy = grad(dr_dx, 1)

        # generate random values for x, y and c
        random_x = np.random.randint(1, 100, 10).astype(float)
        random_y = np.random.randint(1, 100, 10).astype(float)
        random_c = np.random.randint(10, 100, 10).astype(float)

        # check if the random values violate convexity
        for x, y, c in zip(random_x, random_y, random_c):
            #c = float(np.random.randint(max(x,y), 100))
            if (dr2_dx2(x, y, c) < 0):
                print("second derivative wrt x negative x=", \
                        x, " y = ",  y, "c = ", c, "value is", dr2_dx2(x,y,c))
            if (dr2_dy2(x, y, c) < 0):
                print("second derivative wrt y negative x=", \
                        x, " y = ",  y, "c = ", c, "value is", dr2_dy2(x,y,c))
            if (determinant(x, y, c, dr2_dx2, dr2_dy2, dr2_dxdy) < 0):
                print("determinant is negative x=", \
                        x, " y = ",  y, "c = ", c, "value is", \
                        determinant(x, y, c, dr2_dx2, dr2_dy2, dr2_dxdy))

main()
