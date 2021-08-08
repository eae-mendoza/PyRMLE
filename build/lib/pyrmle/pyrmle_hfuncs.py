import matplotlib.pyplot as plt
import scipy.optimize as scop
import scipy as sp
import random
import numpy as np
import time
import sys
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

class RMLEResult:
    """ Class used to store the result of the rmle() function. 
    Arguments for this class:
    f - NumPy np.array containing all grid-point estimates for \hat{f_\beta}
    
    alpha - floating point that specifies the regularization parameters
    
    alpmth - string that specifies the method of selecting the regularization parameter
    
    T - class object tmatrix
    
    details - meta-data regarding the optimization process
    
    Available methods:
    maxval(self) - returns a list containing the maximum value  of the estimated density \hat{f_\beta} as a floating point,
    its location as a tuple.
    
    mode(self) - returns a list of lists containing possible modes of \hat{f_\beta} as floating points, and its locations as
    tuples.
    
    ev(self) - returns a list of floating points which correspond to the expected value of each beta: [E[B_0], E[B_1],E[B_2]]
    
    cov(self) - returns a d x d np.array which corresponds to the variance-covariance matrix of \hat{f_\beta}.
    
    """
    def __init__(self,f,alpha,alpmth,T,details):
        self.f  = f
        self.Tmat = T.Tmat
        self.T = T
        self.alpha = alpha
        self.alpmth = alpmth
        self.grid = self.T.grid
        self.dim = self.T.grid.dim
        self.f_shaped = shape(self)
        self.details = details
        
    def maxval(self):
        # Extracts the maximum value of \hat{f_\beta}
        mval = np.max(self.f)
        # Get the index of the maximum value of \hat{f_\beta} from its m x 1 np.array-form 
        i = int(np.where(self.f==np.max(self.f))[0])
        # Creates the new intervals which corresponds to center of the individual grid-boxes or cubes.
        B0 = self.grid.b0_grid_points
        B1 = self.grid.b1_grid_points
        B2 = self.grid.b2_grid_points
        
        if self.dim ==2:
            # Converts the index i obtained previously into grid locations based on the 2-dimensional np.array.
            k = self.grid.ks()[0]
            #Gets index based on first axis
            x1=int(np.floor(i/k))
            #Gets index based on second axis
            x2=int(i%k)
            
            if x2 == 0:
                loc=[B0[x1-1],B1[-1]]
                return [mval,loc]
            else: 
                loc=[B0[x1],B1[x2]]
                return [mval,loc]
            
        else:
            # Converts the index i obtained previously into grid locations based on the 3-dimensional np.array.
            k = self.grid.ks()[0]
            k2 = self.grid.ks()[1]
            
            if i > 0:
                #Gets index based on first axis
                x1=int(i%k)
                #Gets index based on second axis
                x2=int(np.floor(int(i%k2)/k))
                #Gets index based on third axis
                x3=int(np.floor(i/k2))
                
                if x2 == 0:
                    loc=[B0[x1-1],B1[-1],B2[x3]]
                    return [mval,loc]
                else: 
                    loc=[B0[x1],B1[x2],B2[x3]]
                    return [mval,loc]
                
            else:
                loc=[B0[0],B1[0],B2[0]]
                return [mval,loc]
            
    def mode(self):
        # Initialize the list to append possible local maxima to
        modes=[]
        B0 = self.grid.b0_grid_points
        B1 = self.grid.b1_grid_points
        B2 = self.grid.b2_grid_points
        dim = self.grid.dim
        
        if dim == 2:
            k = self.grid.ks()[0]
            # Index adjustments to get neighboring points for a 2-dimensional np.array
            neighbor_ks = [-1,1,-k,-(k+1),-(k-1),k,k-1,k+1]
            
            for i in range(0,len(self.f)):
                neighbors = []
                for j in neighbor_ks:
                # The try / pass here is necessary in case f[i] does not have
                # a neighboring point in some directions
                    try:
                        neighbors.append(self.f[i+j])
                    except:
                        pass
                neighbors = np.array(neighbors)
                # Check if there are any neighboring values to f[i] that are greater
                val = sum(neighbors > self.f[i])
                
                if val == 0: 
                    # This case happens when there are no neighboring points that are greater
                    # Which means f[i] is a local maximum
                    if i > 0:
                        #Gets index based on first axis
                        x1=int(np.floor(i/k))
                        #Gets index based on second axis
                        x2=int(i%k)
                        if x2 == 0:
                            loc=[B0[x1-1],B1[-1]]
                            modes.append([self.f[i],loc])
                        else: 
                            loc=[B0[x1],B1[x2]]
                            modes.append([self.f[i],loc])
                            
                    else:
                        loc=[B0[0],B1[0]]
                        modes.append([self.f[i],loc])
                        
            modes.sort(reverse=True)
            
            return modes[:6]
        
        else:
            k = self.grid.ks()[0]
            k2 = self.grid.ks()[1]
            # Index adjustments to get neighboring points for a 3-dimensional np.array
            neighbor_ks = [-1,1,-k,-(k+1),-(k-1),k,k-1,k+1,-1-k2,1-k2,-k-k2,-(k+1)-k2,-(k-1)-k2,k-k2,k-1-k2,k+1-k2,-1+k2,1+k2,-k+k2,-(k+1)+k2,-(k-1)+k2,k+k2,k-1+k2,k+1+k2]
            
            for i in range(0,len(self.f)):
                neighbors = []
                for j in neighbor_ks:
                    try:
                        neighbors.append(self.f[i+j])
                    except:
                        pass
                neighbors = np.array(neighbors)
                # Check if there are any neighboring values to f[i] that are greater
                val = sum(neighbors > self.f[i])
                
                if val == 0:
                    if i > 0:
                        #Gets index based on first axis
                        x1=int(i%k)
                        #Gets index based on second axis
                        x2=int(np.floor(int(i%k2)/k))
                        #Gets index based on third axis
                        x3=int(np.floor(i/k2))
                        if x2 == 0:
                            loc=[B0[x1-1],B1[-1],B2[x3]]
                            modes.append([self.f[i],loc])
                        else: 
                            loc=[B0[x1],B1[x2],B2[x3]]
                            modes.append([self.f[i],loc])
                    else:
                        loc=[B0[0],B1[0],B2[0]]
                        modes.append([self.f[i],loc])
                        
            modes.sort(reverse=True)
            return modes[:6]
        
    def ev(self):
        # Creates the new intervals which corresponds to center of the individual grid-boxes or cubes.
        B0 = (self.grid.b0_grid_points+self.grid.shifts[0])*self.grid.b0
        B1 = (self.grid.b1_grid_points+self.grid.shifts[1])*self.grid.b0
        B2 = (self.grid.b2_grid_points+self.grid.shifts[2])*self.grid.b0
        # Initialize the list for expected values
        expected_vals = []
        m = self.grid.ks()[0]
        # Call \hat{f_\beta}
        f = self.f
        # Declare the step size of the grid based on the grid used to create the transformation matrix
        b0_step = B0[1]-B0[0]
        b1_step = B1[1]-B1[0]
        b2_step = B2[1]-B2[0]
        
        if self.dim == 2:
            # Reshape \hat{f_\beta} to the it's 2-dimensional np.array form
            f_shaped = f.reshape(m,m)
            # E[B0] = \int{}
            expected_vals.append(sum(np.sum(f_shaped,axis=1)*B0*b0_step*b1_step))
            expected_vals.append(sum(np.sum(f_shaped,axis=0)*B1*b0_step*b1_step))
            return shift_scale_loc(self.grid,expected_vals)
        
        elif self.dim == 3:
            f_shaped = f.reshape(m,m,m)
            f12=np.sum(f_shaped,axis=2)*b0_step
            f02=np.sum(f_shaped,axis=0)*b1_step
            f01=np.sum(f_shaped,axis=1)*b2_step
            expected_vals.append(sum(np.sum(f01,axis=0)*B0*b0_step*b2_step))
            expected_vals.append(sum(np.sum(f02,axis=1)*B1*b1_step*b2_step))
            expected_vals.append(sum(np.sum(f12,axis=1)*B2*b2_step*b0_step))
            return shift_scale_loc(self.grid,expected_vals)
        
    def cov(self):
        # Establish how many dimensions \hat{f_\beta} has
        dim = self.dim
        # Call \hat{f_\beta}
        f = self.f
        m = self.grid.ks()[0]
        # Set gridpoints for each axis
        B0 = self.grid.b0_grid_points
        B1 = self.grid.b1_grid_points
        B2 = self.grid.b2_grid_points
        # Set the step size of each axis
        b0_step = self.grid.step/self.grid.b0
        b1_step = self.grid.step/self.grid.b1
        b2_step = self.grid.step/self.grid.b2
        
        if dim ==2:
            # Reshape \hat{f_\beta} to 2-dimensional form
            shaped = f.reshape(m,m)
            # Initialize the covariance matrix
            cov_mat = np.zeros((2,2))
            # Compute for the marginal distributions
            f0=shaped.sum(axis=1)*b0_step
            f1=shaped.sum(axis=0)*b1_step
            # Assign the variances on the diagonal entries
            cov_mat[0][0] = sum(f0*(B0-self.ev()[0])**2*b0_step)
            cov_mat[1][1] = sum(f1*(B1-self.ev()[1])**2*b1_step)
            # Extract the values of \hat{f_\beta} for which b_0 = b_1
            f01 = []
            
            for i in range(0,self.grid.ks()[0]):
                f01.append(shaped[i][i])
            f01 = np.array(f01)
            # Compute for the covariance
            val = sum(f01*(B0-self.ev()[0])*(B1-self.ev()[1])*b0_step*b1_step)
            cov_mat[0][1],cov_mat[1][0] = val,val
            return cov_mat
        
        else:
            # Reshape \hat{f_\beta} to its 3-dimensional form
            shaped = f.reshape(m,m,m)
            # INitialize the covariance matrix
            cov_mat = np.zeros((3,3))
            # Compute for the marginal distributions
            f12=np.sum(shaped,axis=2)*b0_step
            f02=np.sum(shaped,axis=0)*b1_step
            f01=np.sum(shaped,axis=1)*b2_step
            f0 = np.sum(f01,axis=0)*b1_step
            f1 = np.sum(f12,axis=0)*b2_step
            f2 = np.sum(f02,axis=1)*b0_step
            # Compute for the variances on the diagonal entries
            var_00 = sum(f0*(B0-self.ev()[0])**2*b0_step**2)
            var_11 = sum(f1*(B1-self.ev()[1])**2*b1_step**2)
            var_22 = sum(f2*(B2-self.ev()[2])**2*b2_step**2)
            cov_mat[0][0], cov_mat[1][1], cov_mat[2][2] = var_00, var_11, var_22
            # Extract the values from the appropriate marginal distributions 
            f01s,f02s, f12s = [], [], []
            
            for i in range(0,grid_est.ks()[0]):
                f01s.append(f01[i][i])
                f02s.append(f02[i][i])
                f12s.append(f12[i][i])
            f01s = np.array(f01s)
            f02s = np.array(f02s)
            f12s = np.array(f12s)
            # Compute for the covariances
            var01 = np.sum(f01s*(B0-self.ev()[0])*(B1-self.ev()[1])*b0_step*b1_step)
            var02 = np.sum(f02s*(B0-self.ev()[0])*(B2-self.ev()[2])*b0_step*b2_step)
            var12 = np.sum(f12s*(B1-self.ev()[1])*(B2-self.ev()[2])*b1_step*b2_step)
            cov_mat[0][1], cov_mat[1][0] = var01, var01
            cov_mat[0][2], cov_mat[2][0] = var02, var02
            cov_mat[1][2], cov_mat[2][1] = var12, var12
            return cov_mat        
        
class tmatrix:
    """ Class that stores the output of the transmatrix() function. 
    Attributes/arguments for this class:
    
    Tmat - The transformation matrix \mathbf{T} used in the optimzation algorithm, which is generated from 
    the sample observations.
    
    grid - Returns the grid class which was used to generate the transformation matrix
    
    scaled_sample -  Returns the scaled and shifted sample. This will return the original sample if the user 
    chooses the default grid range. Otherwise, it will return a linear transformation of the sample that reflects
    the change in location and scale of the grid.
    
    sample - Returns the original sample.
    
    Availabale methods:
    
    n() - Method that returns the number of rows in Tmat, which corresponds to the sample size.
    
    m() - Method that returns the number of columns in Tmat, which corresponds to the number of grid points.
    """
    def __init__(self,Tmat,grid,scaled_sample,sample):
        self.Tmat  = Tmat
        self.grid = grid
        self.scaled_sample = scaled_sample
        self.sample = sample
        
    def n(self):
        return len(self.Tmat)
    
    def m(self):
        return len(self.Tmat.T)
    
class grid_obj: 
    """This class is the output of the grid_set() function.
    Arguments of this class:
    
    scale - np.array of constants that reflect the change in scale of the total distance covered by the grid in reference
    to the base range of [-5,5] with a interval length of 10.
    
    shifts - np.array of constants that reflect the change in location of the center of the grid. The base grid is centered 
    at the origin.
    
    interval - A list that is converted into a NumPy np.array, that represents an axis of the base grid over which 
    \hat{f_\beta} is estimated over.
    
    dim - An integer that specifies the dimensionality of the problem.
    
    step - A float that indicates the step size of the base grid.
    
    start - A float that indicates the minimum value of the grid.
     
    end - A float that indicates the maximum of the grid.
    """
    def __init__(self,scale,shifts,interval,dim,step,start,end):
        self.interval = np.array(interval)
        self.dim = dim
        self.step = step
        self.start = start
        self.end = end
        self.scale = scale
        self.b0 = scale[0]
        self.b1 = scale[1]
        self.b2 = scale[2]
        # Computes for the scaled and shifted axes
        self.b0_range =(interval)/scale[0]-np.array(shifts[0])
        self.b1_range =(interval)/scale[1]-np.array(shifts[1])
        self.b2_range =(interval)/scale[2]-np.array(shifts[2])
        # Computes for grid-points over which \hat{f_\beta} is estimated over. The actual grid points are not the
        # points in the axis ranges, but the mid-points of each axis since the method estimates the value of 
        # \hat{f_\beta} at the middle of the grid boxes.
        self.b0_grid_points = np.array([(self.b0_range[i-1]+self.b0_range[i])/2 for i in range(1,len(self.b0_range))])
        self.b1_grid_points = np.array([(self.b1_range[i-1]+self.b1_range[i])/2 for i in range(1,len(self.b1_range))])
        self.b2_grid_points = np.array([(self.b2_range[i-1]+self.b2_range[i])/2 for i in range(1,len(self.b2_range))])
        self.shifts = shifts
        self.scaled_steps = step/scale
        
    def ks(self):
        k = []
        m = len(self.interval)
        
        for i in range(0,self.dim):
            k.append(int((m-1)**(i+1)))
        return k
    
    def numgridpoints(self):
        return (len(self.interval)-1)**self.dim

def transmatrix(xy_sample,grid):
    """ This function just serves as a wrapper function for the 2-d and 3-d implementations of the algorithm
    used to generate the transformation matrix.
    """
    scaled_sample = scale_sample(xy_sample,grid)
    sample = xy_sample.copy()
    samples = sample_obj(scaled_sample=scaled_sample,sample=sample)
    
    if grid.dim == 2:
        tmatrix = transmatrix_2d(samples,grid)
        return tmatrix
    
    else:
        tmatrix = transmatrix_3d(samples,grid)
        return tmatrix

class SplineResult:
    def __init__(self, f,f_shaped,joint_marginals,spline_grid):
        self.f = f
        self.f_shaped = f_shaped
        self.joint_marginals = joint_marginals
        self.grid = spline_grid
        self.dim = spline_grid.dim
        self.num_grid_points = spline_grid.num_grid_points
        
    def maxval(self):
        # Extracts the maximum value of \hat{f_\beta}
        mval = np.max(self.f)
        # Get the index of the maximum value of \hat{f_\beta} from its m x 1 np.array-form
        i = int(np.where(self.f == np.max(self.f))[0])
        # Creates the new intervals which corresponds to center of the individual grid-boxes or cubes.
        B0 = self.grid.b0_grid_points
        B1 = self.grid.b1_grid_points
        B2 = self.grid.b2_grid_points
        
        if self.dim == 2:
            # Converts the index i obtained previously into grid locations based on the 2-dimensional np.array.
            k = self.grid.ks()[0]
            # Gets index based on first axis
            x1 = int(np.floor(i / k))
            # Gets index based on second axis
            x2 = int(i % k)
            if x2 == 0:
                loc = [B0[x1 - 1], B1[-1]]
                return [mval, loc]
            else:
                loc = [B0[x1], B1[x2]]
                return [mval, loc]
            
        else:
            # Converts the index i obtained previously into grid locations based on the 3-dimensional np.array.
            k = self.grid.ks()[0]
            k2 = self.grid.ks()[1]
            if i > 0:
                # Gets index based on first axis
                x1 = int(i % k)
                # Gets index based on second axis
                x2 = int(np.floor(int(i % k2) / k))
                # Gets index based on third axis
                x3 = int(np.floor(i / k2))
                if x2 == 0:
                    loc = [B0[x1 - 1], B1[-1], B2[x3]]
                    return [mval, loc]
                else:
                    loc = [B0[x1], B1[x2], B2[x3]]
                    return [mval, loc]
                
            else:
                loc = [B0[0], B1[0], B2[0]]
                return [mval, loc]

    def mode(self):
        # Initialize the list to append possible local maxima to
        modes = []
        B0 = self.grid.b0_grid_points
        B1 = self.grid.b1_grid_points
        B2 = self.grid.b2_grid_points
        dim = self.grid.dim
        
        if dim == 2:
            k = self.grid.ks()[0]
            # Index adjustments to get neighboring points for a 2-dimensional np.array
            neighbor_ks = [-1, 1, -k, -(k + 1), -(k - 1), k, k - 1, k + 1]
            
            for i in range(0, len(self.f)):
                neighbors = []
                for j in neighbor_ks:
                    # The try / pass here is necessary in case f[i] does not have
                    # a neighboring point in some directions
                    try:
                        neighbors.append(self.f[i + j])
                    except:
                        pass
                neighbors = np.array(neighbors)
                # Check if there are any neighboring values to f[i] that are greater
                val = sum(neighbors > self.f[i])
                
                if val == 0:
                    # This case happens when there are no neighboring points that are greater
                    # Which means f[i] is a local maximum
                    if i > 0:
                        # Gets index based on first axis
                        x1 = int(np.floor(i / k))
                        # Gets index based on second axis
                        x2 = int(i % k)
                        if x2 == 0:
                            loc = [B0[x1 - 1], B1[-1]]
                            modes.append([self.f[i], loc])
                        else:
                            loc = [B0[x1], B1[x2]]
                            modes.append([self.f[i], loc])
                            
                    else:
                        loc = [B0[0], B1[0]]
                        modes.append([self.f[i], loc])
            modes.sort(reverse=True)
            
            return modes[:6]
        
        else:
            k = self.grid.ks()[0]
            k2 = self.grid.ks()[1]
            # Index adjustments to get neighboring points for a 3-dimensional np.array
            neighbor_ks = [-1, 1, -k, -(k + 1), -(k - 1), k, k - 1, k + 1, -1 - k2, 1 - k2, -k - k2, -(k + 1) - k2,
                           -(k - 1) - k2, k - k2, k - 1 - k2, k + 1 - k2, -1 + k2, 1 + k2, -k + k2, -(k + 1) + k2,
                           -(k - 1) + k2, k + k2, k - 1 + k2, k + 1 + k2]
            
            for i in range(0, len(self.f)):
                neighbors = []
                for j in neighbor_ks:
                    try:
                        neighbors.append(self.f[i + j])
                    except:
                        pass
                neighbors = np.array(neighbors)
                # Check if there are any neighboring values to f[i] that are greater
                val = sum(neighbors > self.f[i])
                
                if val == 0:
                    if i > 0:
                        # Gets index based on first axis
                        x1 = int(i % k)
                        # Gets index based on second axis
                        x2 = int(np.floor(int(i % k2) / k))
                        # Gets index based on third axis
                        x3 = int(np.floor(i / k2))
                        if x2 == 0:
                            loc = [B0[x1 - 1], B1[-1], B2[x3]]
                            modes.append([self.f[i], loc])
                        else:
                            loc = [B0[x1], B1[x2], B2[x3]]
                            modes.append([self.f[i], loc])
                            
                    else:
                        loc = [B0[0], B1[0], B2[0]]
                        modes.append([self.f[i], loc])
                        
            modes.sort(reverse=True)
            return modes[:6]

    def ev(self):
        # Creates the new intervals which corresponds to center of the individual grid-boxes or cubes.
        B0 = (self.grid.b0_grid_points+self.grid.shifts[0])*self.grid.b0
        B1 = (self.grid.b1_grid_points+self.grid.shifts[1])*self.grid.b1
        B2 = (self.grid.b2_grid_points+self.grid.shifts[2])*self.grid.b2
        # Initialize the list for expected values
        expected_vals = []
        m = self.grid.ks()[0]
        # Call \hat{f_\beta}
        f = self.f
        # Declare the step size of the grid based on the grid used to create the transformation matrix
        b0_step = B0[1]-B0[0]
        b1_step = B1[1]-B1[0]
        b2_step = B2[1]-B2[0]
        
        if self.dim == 2:
            # Reshape \hat{f_\beta} to the it's 2-dimensional np.array form
            f_shaped = self.f_shaped
            # E[B0] = \int{}
            expected_vals.append(sum(np.sum(f_shaped, axis=1) * B0 * b0_step * b1_step))
            expected_vals.append(sum(np.sum(f_shaped, axis=0) * B1 * b0_step * b1_step))
            return shift_scale_loc(self.grid,expected_vals)
        
        elif self.dim == 3:
            f12 = self.joint_marginals[2]
            f02 = self.joint_marginals[1]
            f01 = self.joint_marginals[0]
            expected_vals.append(sum(np.sum(f01, axis=0) * B0 * b0_step * b2_step))
            expected_vals.append(sum(np.sum(f02, axis=1) * B1 * b1_step * b2_step))
            expected_vals.append(sum(np.sum(f12, axis=1) * B2 * b2_step * b0_step))
            return shift_scale_loc(self.grid,expected_vals)
        
    def cov(self):
        # Establish how many dimensions \hat{f_\beta} has
        dim = self.dim
        # Call \hat{f_\beta}
        f = self.f
        m = self.grid.ks()[0]
        # Set gridpoints for each axis
        B0 = self.grid.b0_grid_points
        B1 = self.grid.b1_grid_points
        B2 = self.grid.b2_grid_points
        # Set the step size of each axis
        b0_step = B0[1]-B0[0]
        b1_step = B1[1]-B1[0]
        b2_step = B2[1]-B2[0]
        
        if dim == 2:
            # Reshape \hat{f_\beta} to 2-dimensional form
            shaped = self.f_shaped
            # Initialize the covariance matrix
            cov_mat = np.zeros((2, 2))
            # Compute for the marginal distributions
            f0 = shaped.sum(axis=1) * b0_step
            f1 = shaped.sum(axis=0) * b1_step
            # Assign the variances on the diagonal entries
            cov_mat[0][0] = sum(f0 * (B0 - self.ev()[0]) ** 2 * b0_step)
            cov_mat[1][1] = sum(f1 * (B1 - self.ev()[1]) ** 2 * b1_step)
            # Extract the values of \hat{f_\beta} for which b_0 = b_1
            f01 = []
            
            for i in range(0, self.grid.ks()[0]):
                f01.append(shaped[i][i])
            f01 = np.array(f01)
            # Compute for the covariance
            val = sum(f01 * (B0 - self.ev()[0]) * (B1 - self.ev()[1]) * b0_step * b1_step)
            cov_mat[0][1], cov_mat[1][0] = val, val
            return cov_mat
        
        else:
            # INitialize the covariance matrix
            cov_mat = np.zeros((3, 3))
            # Compute for the marginal distributions
            f12 = self.joint_marginals[2]
            f02 = self.joint_marginals[1]
            f01 = self.joint_marginals[0]
            f0 = np.sum(f01, axis=0) * b1_step
            f1 = np.sum(f12, axis=0) * b2_step
            f2 = np.sum(f02, axis=1) * b0_step
            # Compute for the variances on the diagonal entries
            var_00 = sum(f0 * (B0 - self.ev()[0]) ** 2 * b0_step ** 2)
            var_11 = sum(f1 * (B1 - self.ev()[1]) ** 2 * b1_step ** 2)
            var_22 = sum(f2 * (B2 - self.ev()[2]) ** 2 * b2_step ** 2)
            cov_mat[0][0], cov_mat[1][1], cov_mat[2][2] = var_00, var_11, var_22
            # Extract the values from the appropriate marginal distributions
            f01s, f02s, f12s = [], [], []
            
            for i in range(0, self.grid.ks()[0]):
                f01s.append(f01[i][i])
                f02s.append(f02[i][i])
                f12s.append(f12[i][i])
            f01s = np.array(f01s)
            f02s = np.array(f02s)
            f12s = np.array(f12s)
            # Compute for the covariances
            var01 = np.sum(f01s * (B0 - self.ev()[0]) * (B1 - self.ev()[1]) * b0_step * b1_step)
            var02 = np.sum(f02s * (B0 - self.ev()[0]) * (B2 - self.ev()[2]) * b0_step * b2_step)
            var12 = np.sum(f12s * (B1 - self.ev()[1]) * (B2 - self.ev()[2]) * b1_step * b2_step)
            cov_mat[0][1], cov_mat[1][0] = var01, var01
            cov_mat[0][2], cov_mat[2][0] = var02, var02
            cov_mat[1][2], cov_mat[2][1] = var12, var12
            
            return cov_mat

class spline_grid_obj:
    
    def __init__(self, num_grid_points,b0_grid_points,b1_grid_points,b2_grid_points,dim,scale,shifts):
        self.num_grid_points = num_grid_points
        self.b0_grid_points = b0_grid_points
        self.b1_grid_points = b1_grid_points
        self.b2_grid_points = b2_grid_points
        self.dim = dim
        self.scale = scale
        self.b0 = scale[0]
        self.b1 = scale[1]
        self.b2 = scale[2]
        self.shifts = shifts
        
    def ks(self):
        k = []
        for i in range(0,self.dim):
            k.append(self.num_grid_points**(i+1))
        return k

def ransample_bivar(n,pi,mu,sigma):
    """ Function used to generate \betas from a normal mixture """
    x=np.zeros((n))
    y=np.zeros((n))
    # This step determines from which distribution to sample from based on the 
    # probabilities given
    k=np.random.choice(len(pi),n,p=pi,replace=True)
    
    for i in range(0,len(k)):
        x[i],y[i]=np.random.multivariate_normal(mu[k[i]],sigma[k[i]],1).T
    return x,y

def shape(result):
    f = result.f
    dim = result.dim
    m = result.grid.ks()[0]
    
    if dim  == 2:
        return f.reshape(m,m)
    else:
        return f.reshape(m,m,m)
    
def initial_gauss(grid):
    """ This function generates an initial guess as a gaussian distribution with mean 0 and variance 1 """
    dim = grid.dim
    x_ = grid.b0_grid_points
    y_ = grid.b1_grid_points
    z_ = grid.b2_grid_points
    sig = 1
    mu = 0
    
    if dim == 2:
        x,y = np.meshgrid(x_, y_)
        dst = np.sqrt(x*x+y*y)
        gauss = np.exp(-( (dst-mu)**2 / ( 2.0 * sig**2 ) ) )
        return np.ravel(gauss)
    
    elif dim == 3:
        x,y,z = np.meshgrid(x_, y_, z_, indexing='xy')
        dst = np.sqrt(x*x+y*y+z*z)
        gauss = np.exp(-( (dst-mu)**2 / ( 2.0 * sig**2 ) ) )
        return np.ravel(gauss)

def sim_sample2d(n,x_params=None,beta_pi=None,beta_mu=None,beta_cov=None):
    # Checks if user declares any parameters to use.
    if x_params is not None:
        x_params = x_params
    else:
        x_params = [-2,2]
    if beta_pi is not None:
        beta_pi = beta_pi
    else:
        beta_pi = [0.5,0.5]
    if beta_mu is not None:
        beta_mu = beta_mu
    else:
        beta_mu = [[-0.5,-0.5],[0.5,0.5]]
    if beta_cov is not None:
        beta_cov = beta_cov 
    else:
        beta_cov = [[[0.01, 0], [0, 0.01]],[[0.01, 0], [0, 0.01]]]
        
    # Sample the regressor X_1 from a uniform distribution
    x1=np.random.uniform(x_params[0],x_params[1],n).T
    # X_0 is generated by creating a column of ones
    x0=np.repeat(1,n)
    # Create an array of regressors
    xs = np.c_[x0,x1]
    # Sample the random coefficients from a bivariate mixture 
    # Using the function ransample_bivar
    b0,b1=ransample_bivar(n,beta_pi,beta_mu,beta_cov)
    b={'col1':b0,'col2':b1}
    bs = np.c_[b0,b1]
    # Computes for B'*X
    bx=bs*xs
    # Generate the response variable Y 
    y=np.array([sum(x) for x in bx])
    xy_sample_sim=np.c_[xs,y]
    return xy_sample_sim

def sim_sample3d(n,x_params=None,beta_pi=None,beta_mu=None,beta_cov=None):
    # Checks if user declares any parameters to use.
    if x_params is not None:
        x_params = x_params
    else:
        x_params = [-2,2]
    if beta_pi is not None:
        beta_pi = beta_pi
    else:
        beta_pi = [1,0,0]
    if beta_mu is not None:
        beta_mu = beta_mu
    else:
        beta_mu = [[2,2,2],[2,2,2],[2,2,2]]
    if beta_cov is not None:
        beta_cov = beta_cov 
    else:
        beta_cov = [[[0.01, 0, 0], [0,0.01,0], [0,0,0.01]],[[0.01, 0, 0], [0,0.01,0], [0,0,0.01]],[[0.01, 0, 0], [0,0.01,0], [0,0,0.01]]]
        
    # Sample the regressors X_1, and X_2 iid from a uniform distribution.
    x1=np.random.uniform(x_params[0],x_params[1],n).T
    x2=np.random.uniform(x_params[0],x_params[1],n).T
    x0=np.repeat(1,n)
    #creating the np.array of Xs
    xs = np.c_[x0,x1,x2]
    # Sample the random coefficients from a gaussian mixture distribution using the
    # function ransample()
    b0,b1,b2=ransample(n,beta_pi,beta_mu,beta_cov)
    bs = np.c_[b0,b1,b2]
    bx=bs*xs
    # Generate the response variable Y
    y=np.array([sum(x) for x in bx])
    xy_sample_sim=np.c_[xs,y]
    
    return xy_sample_sim


def filt(start,end,step,array):
    """ This function filters out \beta values that exceed the range of
    the grid. 
    """
    t=array[:,0]>=start
    array=array[t]
    t1=array[:,0]<=end-step
    array=array[t1]
    t2=array[:,1]>=start
    array=array[t2]
    t3=array[:,1]<=end-step
    array=array[t3]
    return array


def filt2(array,slope):
    n=len(array)
    if (slope[1]<=0 and slope[0]>=0) or (slope[1]>=0 and slope[0]<=0):
        return array[1:n]
    else:
        return array[1:n]

def which_Bi(point,Bi_grid):
    if len(b1s[b1s<point[0]])>0:
        i=len(b1s[b1s<point[0]])-1
    else:
        i=0
    if len(b0s[b0s<point[1]])>0:
        j=len(b0s[b0s<point[1]])-1
    else:
        j=0
    return Bi_grid[i][j]

def which_ij(point,slope,interval):
    """This function takes an intersection point as an argument and determines the
    2-dimensional index of the intersection point based on the grid. """
    b0s=interval
    b1s=interval
    
    if (slope[1]<=0 and slope[0]>=0) or (slope[1]>=0 and slope[0]<=0):
        if len(b1s[b1s<point[0]])>0: 
            i=len(b1s[b1s<point[0]])-1
        else:
            i=0
        if len(b0s[b0s<point[1]])>0:
            j=len(b0s[b0s<point[1]])-1
        else:
            j=0
        return i,j
    
    else:
        if len(b1s[b1s<point[0]])>0: 
            i=len(b1s[b1s<point[0]])-1
        else:
            i=0
        if len(b0s[b0s<=point[1]])>0:
            j=len(b0s[b0s<=point[1]])-1
        else:
            j=0
        return i,j

def index_conv(ijs,k):
    """ This function converts the indices i and j, obtained from the function
    which_ij() into the index of the 1 x m dimensional array form of \hat{f_\beta}
    """
    val=np.array([x[1]*np.sqrt(k)+x[0] for x in ijs])
    return val

def get_intervals(xi,yi,grid):
    """ This function computes the length of intersection of the line parametrized by the sample 
    observations X and Y with the grid (i.e. returns all the lengths of the line-segments generated
    by passing a line through the grid)
    """
    start = grid.start
    end = grid.end
    step = grid.step
    b0s=grid.interval
    b1s=grid.interval
    # Computes for intersection points for different cases
    
    if xi[1]!= 0 and xi[0]!=0:
        b0=(yi-b1s*xi[1])/xi[0]
        b1=(yi-b0s*xi[0])/xi[1]
        
    elif xi[1]==0:
        b0=[yi]*len(b1s)
        b1=b1s
        
    else:
        b0=b0s
        b1=[yi]*len(b1s)
    b1based=np.c_[b1s,b0]
    b0based=np.c_[b1,b0s]
    b1b0=np.r_[b1based,b0based]
    b1b0=b1b0[np.argsort(b1b0[:,0])]
    # Filters only unique intersections
    b1b0=np.unique(b1b0,axis=0)
    # Filters intersection points within the grid
    new_b1b0=filt(start,end,step,b1b0)
    # Computes the length of the line intervals
    intervals=[np.linalg.norm(new_b1b0[i]-new_b1b0[i+1]) for i in range(0,len(new_b1b0)-1)]
    return intervals

def get_intersections(xi,yi,grid):
    """ This function computes for the intersection points on the grid.
    """
    start = grid.start
    end = grid.end
    step = grid.step
    b0s=grid.interval
    b1s=grid.interval
    
    if xi[1]!= 0 and xi[0]!=0:
        b0=(yi-b1s*xi[1])/xi[0]
        b1=(yi-b0s*xi[0])/xi[1]
        
    elif xi[1]==0:
        b0=[yi]*len(b1s)
        b1=b1s
        
    else:
        b0=b0s
        b1=[yi]*len(b1s)
    b1based=np.c_[b1s,b0]
    b0based=np.c_[b1,b0s]
    b1b0=np.r_[b1based,b0based]
    b1b0=b1b0[np.argsort(b1b0[:,0])]
    b1b0=np.unique(b1b0,axis=0)
    new_b1b0=filt(start,end,step,b1b0)
    reduced_b1b0=filt2(new_b1b0,xi)
    return reduced_b1b0

def transmatrix_2d(sample,grid):
    """ This function is used to generate the transformation matrix for the 2-dimensional
    application.
    
    The output is the class tmatrix which is used as an argument in the wrapper function
    rmle().
    """
    xy_sample = sample.scaled_sample.copy()
    L=np.zeros((len(xy_sample),grid.numgridpoints()))
    
    for n in range(0,len(xy_sample)):
        intervals=get_intervals(xy_sample[n],xy_sample[n][2],grid)
        intersection=get_intersections(xy_sample[n],xy_sample[n][2],grid)
        indices=index_conv([which_ij(p,xy_sample[n],grid.interval) for p in intersection],grid.numgridpoints())
        indices=list(map(int,indices))
        
        for i in range(0,len(indices)):
            L[n][indices[i]]=intervals[i]
    L = L[~np.all(L==0, axis=1)]
    
    return tmatrix(Tmat=L,grid=grid,scaled_sample=xy_sample,sample=sample.sample)


def likelihood(f,n,L_mat_long):
    """ This function is the log-likelihood functional without a regularization 
    term.
    """
    L_mat=L_mat_long.reshape(n,len(f))
    f[f<0]=1e-6
    val=np.log(np.dot(L_mat,f))
    return -sum(val)/n

def norm_sq(f,alpha,n,L_mat_long,step):
    """ This function is the log-likelihood functional with the squared L2 norm 
    of \hat{f_\beta} as the regularization term.
    """
    L_mat=L_mat_long.reshape(n,len(f))
    f[f<0]=1e-6
    val=np.log(np.dot(L_mat,f))
    return -sum(val)/n+ alpha*step**2*sum(f**2)

def sobolev(f,alpha,n,L_mat_long,step):
    """ This function is the log-likelihood functional with the sobolev norm 
    for H1 as the regularization term.
    """
    L_mat=L_mat_long.reshape(n,len(f))
    f[f<0]=1e-6
    val=-np.sum(np.log(np.dot(L_mat,f)))/n
    penal=alpha*step**2*sum(f**2)+alpha*step**2*norm_fprime(f,step)
    return val + penal

def entropy(f,alpha,n,L_mat_long,step):
    """ This function is the log-likelihood functional with the entropy of \hat{f_\beta} 
    as the regularization term.
    """
    L_mat=L_mat_long.reshape(n,len(f))
    f[f<0]=1e-6
    val=np.log(np.dot(L_mat,f))
    return -sum(val)/n + alpha*step**2*sum(f*np.log(f))

def tot_deriv(f,step):
    """ This function comutes the total derivative of f"""
    f=f.reshape(int(np.sqrt(len(f))),int(np.sqrt(len(f))))
    fgrad=np.gradient(f,step)
    return np.ravel((np.sqrt(fgrad[0]**2+fgrad[1]**2)))

def norm_fprime(f,step):
    """ This function computes d/df (||f||^{2}_{2}). """
    f=f.reshape(int(np.sqrt(len(f))),int(np.sqrt(len(f))))
    fgrad=np.gradient(f,step)
    return sum(np.ravel((fgrad[0]**2+fgrad[1]**2)))

def second_deriv(f,step):
    """ This function computes the Laplacian of f."""
    f=f.reshape(int(np.sqrt(len(f))),int(np.sqrt(len(f))))
    fgrad0=np.ravel(np.gradient(np.gradient(f,step)[0],step)[0])
    fgrad1=np.ravel(np.gradient(np.gradient(f,step)[1],step)[1])
    return fgrad0+fgrad1

def jac_likelihood(f,n,L_mat_long):
    """ This function computes the jacobian of the regularization
    functional without any penalty term.
    """
    f[f<0]=1e-6
    L_mat=L_mat_long.reshape(n,len(f))
    denom=np.dot(L_mat,f)
    val=L_mat.T/denom
    return -val.T.sum(axis=0)/n


def jac_norm_sq(f,alpha,n,L_mat_long,step):
    """ This function computes the jacobian of the regularization 
    functional with the squared L2 norm penalty.
    """
    f[f<0]=1e-6
    L_mat=L_mat_long.reshape(n,len(f))
    denom=np.dot(L_mat,f)
    val=L_mat.T/denom
    return -val.T.sum(axis=0)/n+alpha*step**2*2*f

def jac_sobolev(f,alpha,n,L_mat_long,step):
    """ This function computes the jacobian of the regularization
    functional with the H1 penalty.
    """
    f[f < 0] = 1e-6
    L_mat=L_mat_long.reshape(n,len(f))
    denom=np.dot(L_mat,f)
    val=L_mat.T/denom
    return -val.T.sum(axis=0)/n+alpha*step**2*2*f-2*alpha*step**2*second_deriv(f,step)

def jac_entropy(f,alpha,n,L_mat_long,step):
    """ This function computes the jacobian of the regularization
    functional with the entropy penalty.
    """
    f[f < 0] = 1e-6
    L_mat=L_mat_long.reshape(n,len(f))
    denom=np.dot(L_mat,f)
    val=L_mat.T/denom
    return -val.T.sum(axis=0)/n+alpha*step**2*(np.log(f)+1)

def ise(f1,f2,step_size):
    val=np.linalg.norm(f1-f2)**2
    return len(f1)**-1*step_size**2*val

def mise(f1,f2,step_size):
    val=np.linalg.norm(f1-f2)**2
    return step_size**2*val

def sq_l2(f1,f2):
    val = np.linalg.norm(f1-f2)**2
    return val

def prop_n(r,q,i):
    """ This function computes for the bounding function used in
    the implementation of Lepkii's balancing principle for selecting
    \alpha. 
    """
    return 8*r**((1-i)/q)

def get_int(x):
    if any(x<0):
        return next(i for i in x if i<=0)
    else:
        return x[-1]
    
def updt(total, progress):

    """
    Displays or updates a console progress bar.
    Source: https://stackoverflow.com/questions/3160699/python-progress-bar
    
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
def updt_cv(total, progress,time_start):

    """
    Displays an indication of the current run-time and max possible iterations.
    
    Modified code from source.
    
    Source: https://stackoverflow.com/questions/3160699/python-progress-bar
    
    """
    time_elapsed = time.time() - time_start
    max_iters, status = total, ""
    
    if progress >= max_iters:
        progress, max_iters = max_iters, "\r\n"
    text = "\r{} / {} possible iterations. Time elapsed: {} seconds. {}".format(progress,total,time_elapsed,status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
def select_cluster(y,kmean_obj,l2s):
    """ This function is used to select which cluster to not discard when implementing
    the shifting algorithm.
    """
    arr_l2 = np.array(l2s)
    clus1 = len(np.where(y==1)[0])
    clus0 = len(np.where(y==0)[0])
    
    if clus1 > clus0:
        return np.where(y==1)[0],1
    
    elif clus0 > clus1:
        return np.where(y==0)[0],0
    
    else:
        clus_center1 = kmean_obj.cluster_centers_[1][0]
        clus_center0 = kmean_obj.cluster_centers_[1][0]
        
        if clus_center1 < clus_center0:
            return np.where(y==1)[0],1
        else: 
            return np.where(y==0)[0],0

def shift_scale_loc(grid,loc):
    """ This fuction takes a location which is an array then backtransforms the
    shift and scaling applied to it in the process of estimation.
    """
    loc_copy = loc.copy()
    if grid.dim == 2:
        loc_copy[0] = loc_copy[0]/grid.b0-grid.shifts[0]
        loc_copy[1] = loc_copy[1]/grid.b1-grid.shifts[1]
        
    else:
        loc_copy[0] = loc_copy[0]/grid.b0-grid.shifts[0]
        loc_copy[1] = loc_copy[1]/grid.b1-grid.shifts[1]
        loc_copy[2] = loc_copy[2]/grid.b2-grid.shifts[2]
        
    return loc_copy


def index_finder(diffs):
    """ This function is used to identify the index of the \alpha value that
    satisfies the Lepskii balancing principle criteria.
    """
    i=0
    
    for d in diffs:
        val = np.sum(d[d<0])
        i+=1
        if val != 0:
            break
            
    return int(i-1)

def prop_n(r,q,i):
    return 8*r**((1-i)/q)

def alpha_vals(a,n):
    """ This function generates the values of \alpha to be used for cross validation. """
    itrs=[]
    
    for i in range(0,n):
        itrs.append(a*0.9**i)
    itrs.reverse()
    
    return itrs

def alpha_lep(n,sample_size,r):
    """ This function generates the values of \alpha to be used for Lepskii's method"""
    a=1/(2*sample_size)*np.log(sample_size)/np.sqrt(sample_size)
    itrs=[]
    
    for i in range(0,n):
        itrs.append(a*r**i)
        
    itrs=[i for i in itrs if i <=2]
    
    return itrs

def sample_shuffle(sample):
    """ This function applies a  random shuffle on the subsamples used in the process of
    cross validation.
    """
    n=len(sample)
    indices=np.arange(0,n)
    random.shuffle(indices)
    i=0
    new_sample=[]
    
    for i in indices:
        new_sample.append(sample[i])
    return new_sample

def cv_index(n,k):
    """ Generates the indices used to slice the sample into the k-folds for cross
    validation.
    """
    n_k = int(np.ceil(n/k))
    
    if n%k != 0:
        indices=np.arange(0,n,n_k)[0:-1]
        
    else:
        indices=np.arange(0,n,n_k)
        
    return indices

def cv_loss(a,alphas,n,k,progress,time_start,total,functional,initial_guess,trans_matrix,trans_matrix_long,step_size,jacobian,hessian_method,constraints,tolerance,max_iter,bound):
    """ This fucntion is used to compute for the loss function in cross-validation. """
    indices=cv_index(n,k)
    f_n=scop.minimize(functional,initial_guess,args=(a,n,trans_matrix_long,step_size),method='trust-constr',jac=jacobian,hess=hessian_method,constraints=[constraints],tol=tolerance,options={'verbose': False,'maxiter': max_iter},bounds=bound)
    fs=[]
    trans_matrix_slices=[]
    inv_trans_matrix_slices=[]
    trans_matrix_slices.append(trans_matrix[indices[1]:])
    trans_matrix_slices.append(trans_matrix[:indices[-1]])
    inv_trans_matrix_slices.append(trans_matrix[:indices[1]])
    inv_trans_matrix_slices.append(trans_matrix[indices[-1]:])
    j=progress
    loss=0
    
    for i in range(1,len(indices)-1):
        trans_matrix_slices.append(np.concatenate((trans_matrix[:indices[i]],trans_matrix[indices[i+1]:])))
        inv_trans_matrix_slices.append(trans_matrix[indices[i]:indices[i+1]])
        
    for t,i in zip(trans_matrix_slices,inv_trans_matrix_slices):
        trans_matrix_slice_long = np.ravel(t)
        inv_trans_matrix_slice_long = np.ravel(i)
        n=len(t)
        n_i = len(i)
        val_f=scop.minimize(functional,initial_guess,args=(a,n,trans_matrix_slice_long,step_size),method='trust-constr',jac=jacobian,hess=hessian_method,constraints=[constraints],tol=tolerance,options={'verbose': False,'maxiter': max_iter},bounds=bound).x
        fs.append(val_f)
        loss+=likelihood_l(val_f,inv_trans_matrix_slice_long,n_i)
        j+=1
        updt_cv(total,j,time_start)
        
    return f_n.x,loss/k,j

def quarter_selector(alphas,n,k,progress,time_start,total,functional,initial_guess,trans_matrix,trans_matrix_long,step_size,jacobian,hessian_method,constraints,tolerance,max_iter,bound):
    """ This function is the implementation of the modified k-fold cross validation method. """
    alphas=alphas
    len_a =len(alphas)
    quart_a = int(np.ceil(len_a)*0.75)
    p,q=random.randint(0,quart_a-1),random.randint(quart_a,len_a-1)
    a_p=alphas[p]
    a_q=alphas[q]
    val1=cv_loss(a_p,alphas,n,k,progress,time_start,total,functional,initial_guess,trans_matrix,trans_matrix_long,step_size,jacobian,hessian_method,constraints,tolerance,max_iter,bound)
    progress_updt = val1[2]
    val2=cv_loss(a_q,alphas,n,k,progress_updt,time_start,total,functional,initial_guess,trans_matrix,trans_matrix_long,step_size,jacobian,hessian_method,constraints,tolerance,max_iter,bound)
    alphas=np.delete(alphas,[p,q])
    
    if val1[1] < val2[1]:
        alphas=alphas[:quart_a-2]
        return val1[0],alphas,val1[1],val2[2],a_p
    else:
        
        alphas=alphas[quart_a-1:len_a-2]
        return val2[0],alphas,val2[1],val2[2],a_q
    
def rmle_2d(functional,alpha,tmat,shift=None,k=None,jacobian=None,initial_guess=None,hessian_method=None,constraints=None,tolerance=None,max_iter=None,bounds=None):
    """ This function is a sub-function of rmle(). This is a wrapper function for the SciPy Optimize minimize()
    function. It has 3 essential arguments which are: {functional, alpha, tmat}. 
    
    Essential/Positional arguments:
    
    functional - the choice of regularization functional
    
    alpha - the choice of the alpha parameter which can be a constant or a string matching 'cv' or 'lepskii'
    
    tmat - the class object tmatrix generated by the the transmatrix() function. This contains the 
    transformation matrix used in the evaluation of the functional and its jacobian.
    
    Optional arguments:
    
    shift - Unused for rmle_2d()
    
    k - an integer value that specifies the number of folds to be used in k-fold cross validation
     
    jacobian - accepts a function as an argument or string type, for more options the please refer to the
    documentation of SciPy's minimize function for more details. In the case of this module, the jacobian of 
    functional's explicit form is provided in the form of a function. We advise to keep this argument to its
    default status.
    
    intiial_guess - an array argument that matches the dimensions of \hat{f_\bbeta}.
     
    hessian_method -  the default is the '2-point' method implemented in the SciPy library. The user can 
    provide the other hessian methods avaiable to the scipy optimize minimize() function.
    
    constraints - the linear constraint for the optimization problem, namely: ||\hat{f_\bbeta}||_L1 = 1 i.e.
    the estimate \hat{f_\bbeta} should integrate to 1.
    
    tolerance - a floating point argument that sets the tolerance value for the optimization problem.
    
    max_iter - an integer that sets the maximum number of iterations.
    
    bounds - sets the bounds for the values of the estimate, the default value is a non-negativity constraint.
    """ 
    st = time.time()
    n=tmat.n()
    m=tmat.m()
    step_size = tmat.grid.step
    trans_matrix_long = np.ravel(tmat.Tmat)
    trans_matrix = tmat.Tmat
    lepskii_matches = ['lep','lepskii','lepskii\'s principle', 'lp']
    cv_matches = ['cv','cross','cross val', 'validation','crossvalidation','cross-validation']
    
    if '3d' in str(functional.__name__):
        raise ValueError("The 2-dimensional implementation requires a matching 2-dimensional functional. \
Try to supply any of the accepted functional values: {likelihood, norm_sq, sobolev, entropy}")
        
    if initial_guess is not None:
        initial_guess = initial_guess
        
    else:
        initial_guess = initial_gauss(tmat.grid)
        
    if not jacobian:
        if functional == likelihood:
            jacobian = jac_likelihood
        elif functional == norm_sq:
            jacobian = jac_norm_sq
        elif functional == sobolev:
            jacobian = jac_sobolev
        elif functional == entropy:
            jacobian =  jac_entropy
            
    else:
        jacobian = jacobian
        
    if not hessian_method:
        hessian_method = '2-point'
        
    else:
        hessian_method = hessian_method
        
    if not constraints:
        linear_constraint=scop.LinearConstraint([step_size**2]*len(initial_guess),[1],[1])
        constraints = linear_constraint
        
    else:
        constraints = constraints
        
    if not tolerance:
        tolerance = 1e-12
        
    else:
        tolerance = tolerance
        
    if not max_iter:
        max_iter = 10*m
        
    else: 
        max_iter = max_iter
        
    if not bounds:
        bound = scop.Bounds(0,np.inf)
        
    else:
        bound = bounds
        
    if not k:
        k = 10
        
    else:
        k = k
        
    if any(c in str.lower(str(alpha)) for c in lepskii_matches):
        r=1.2
        alphas=alpha_lep(70,n,r)
        reconstructions=[]
        total = len(alphas)
        i_range=np.arange(0,len(alphas))
        j=0
        times=[]
        
        for a in alphas:
            rec = scop.minimize(functional,initial_guess,args=(a,n,trans_matrix_long,step_size),method='trust-constr',jac=jacobian,hess=hessian_method,constraints=[constraints],tol=tolerance,options={'verbose': False,'maxiter': max_iter},bounds=bound)
            reconstructions.append(rec.x)
            j+=1
            updt(total,j)
        approx_errs = []
        
        for i in range(1,len(reconstructions)):
            jj=0
            approx_error = []
            while jj < i:
                approx_error.append(np.linalg.norm(reconstructions[i]-reconstructions[jj]))
                jj+=1
            approx_errs.append(approx_error)
        prop_errors=prop_n(r,2,i_range)
        diffs=[]
        
        for p in approx_errs:
            n_p = int(len(p))
            diffs.append(np.array(prop_errors[:n_p])-np.array(p))
            
        index=index_finder(diffs)
        et = time.time()
        details = {
            "Regularization Functional": str(functional.__name__),
            "Jacobian": str(jacobian.__name__),
            "Numer of Grid Points": m,
            "Sample Size": n,
            "Total Run Time": str(et-st) + ' seconds',
            "Numer of Lepskii iterations": len(alphas)
        }
        
        return RMLEResult(f=reconstructions[index],alpha=alphas[index],alpmth='Lepskii',T=tmat,details=details)
    
    if any(c in str.lower(str(alpha)) for c in cv_matches):
        alphas=alpha_vals(step_size*3,30)
        lhood=[]
        reconstructions=[]
        alpha_list=[]
        trans_matrix=sample_shuffle(trans_matrix)
        time_start = time.time()
        j=0
        total = np.ceil((np.log(4)-np.log(len(alphas)))/(0.25*np.log(0.25)+0.75*np.log(0.75)))*4 * k
        
        while len(alphas)> 4:
            val=quarter_selector(alphas,n,k,j,time_start,total,functional,initial_guess,trans_matrix,trans_matrix_long,step_size,jacobian,hessian_method,constraints,tolerance,max_iter,bound)
            lhood.append(val[2])
            reconstructions.append(val[0])
            j=val[3]
            alphas=val[1]
            alpha_list.append(val[4])
            
        for a in alphas:
            val=cv_loss(a,alphas,n,k,j,time_start,total,functional,initial_guess,trans_matrix,trans_matrix_long,step_size,jacobian,hessian_method,constraints,tolerance,max_iter,bound)
            lhood.append(val[1])
            reconstructions.append(val[0])
            alpha_list.append(a)
            j=val[2]
            
        index=np.int(np.where(lhood==np.min(lhood))[0])
        updt_cv(total,total,time_start)
        et = time.time()
        details = {
            "Regularization Functional": str(functional.__name__),
            "Jacobian": str(jacobian.__name__),
            "Numer of Grid Points": m,
            "Sample Size": n,
            "Total Run Time": str(et-st) + ' seconds',
            "Number of CV iterations": j,
        }
        return RMLEResult(f=reconstructions[index],alpha=alpha_list[index],alpmth='CV',T=tmat,details=details)
    
    else:
        result = scop.minimize(functional,initial_guess,args=(alpha,n,trans_matrix_long,step_size),method='trust-constr',jac=jacobian,hess=hessian_method,constraints=[constraints],tol=tolerance,options={'verbose': 0,'maxiter': max_iter},bounds=bound)
        et = time.time()
        details = {
            "Regularization Functional": str(functional.__name__),
            "Jacobian": str(jacobian.__name__),
            "Numer of Grid Points": m,
            "Sample Size": n,
            "Total Run Time": str(et-st) + ' seconds',
            "Iterations": result.niter,
            "Optimization Run Time": result.execution_time,
        }
        
        return RMLEResult(f=result.x,alpha=alpha,alpmth='User',T=tmat,details=details)

def edge_check(p,start,end):
    """ This function checks if the coordinates of a point lies at the edge of a grid. It returns a list of boolean values."""
    check=[]
    
    for i in p:
        check.append((i-end)==0 or (i-start)==0)
    return check
  
def outer_edge_check(p,start,end):
    """ This function checks if a specific coordinate is at the edge of the grid."""
    return (p-end)==0 or (p-start)==0

 
def get_index(point,interval,k):
    """ This function maps an intersection point to its index on the grid i.e. in which box of 
    the grid the point lies in. """
    interval_round=np.array([round(i,8) for i in interval])
    x=[]
    
    for p in point:
        x.append(max(np.where(interval_round>=p)[0][0]-1,0))
        
    index = x[0] + x[1]*k**(1/3) + x[2]*k**(2/3)
    return np.ceil(index)

def point_check(point,start,end):
        """ This function checks if the coordinates of a point lies at the edge of a grid. It returns a list of 
        boolean values. """
        check=0
        
        for i in point:
            if i == start or i == end:
                check+=1
        return check

def vertex_check(point,interval):
    """ This function checks if a point has coordinates that lies on the grid, but are not on outer-edges. """
    val = 0
    
    for p in point:
        if p in interval:
            val+=1
    return val

def get_box_index(point,start,end,step_size,interval,ks):
    """ This function takes a point and returns all the grid boxes (its index) that it lies in. An
    intersection point can lie in more than one box in the grid e.g. if a point lies on a vertex of the
    grid, it lies in eight grid boxes. 
    """
    point=np.array(point)
    if point_check(point,start,end) ==3: #case for outer-vertex
        indices = np.array([get_index(point,interval,ks[2])],dtype=int) 
        return list(indices)
    
    elif ((point_check(point,start,end) == 2) and  vertex_check(point,interval) == 3): 
        #case for corner-outer-edge vertex
        if (abs(point[1])-abs(start)==0 and abs(point[2])-abs(start)==0) or (abs(point[1])-abs(end)==0 and abs(point[2])-abs(end)==0):
            indices = np.array([get_index(point,interval,ks[2])] * 2,dtype=int)
            adjustments = np.array([0,1],dtype=int)
            return list(indices+adjustments)
        
        elif (abs(point[0])-abs(start)==0 and abs(point[2])-abs(start)==0) or (abs(point[0])-abs(end)==0 and abs(point[2])-abs(end)==0):
            indices = np.array([get_index(point,interval,ks[2])] * 2,dtype=int)
            adjustments = np.array([0,ks[0]],dtype=int)
            return list(indices+adjustments) 
        
        elif (abs(point[0])-abs(start)==0 and abs(point[1])-abs(start)==0) or (abs(point[0])-abs(end)==0 and abs(point[1])-abs(end)==0):
            indices = np.array([get_index(point,interval,ks[2])] * 2,dtype=int)
            adjustments = np.array([0,ks[1]],dtype=int)
            return list(indices+adjustments) 
        
    elif ((sum((abs(point)-abs(start))==0) == 1 or sum((abs(point)-abs(end))==0) == 1) and vertex_check(point,interval) == 3): 
        #case for side-outer-edge vertex
        #cases for different axes
        if (abs(point[1])-abs(start)==0) or (abs(point[1])-abs(end)==0):
            indices = np.array([get_index(point,interval,ks[2])] * 4,dtype=int)
            adjustments = np.array([0,1,ks[1],ks[1]+1],dtype=int)
            return list(indices+adjustments)
        
        elif (abs(point[0])-abs(start)==0) or (abs(point[0])-abs(end)==0):
            indices = np.array([get_index(point,interval,ks[2])] * 4,dtype=int)
            adjustments = np.array([0,ks[0],ks[1],ks[1]+ks[0]],dtype=int)
            return list(indices+adjustments)
        
        elif abs(point[2])-abs(start)==0:
            indices = np.array([get_index(point,interval,ks[2])] * 4,dtype=int)
            adjustments = np.array([0,1,ks[0],ks[0]+1],dtype=int)
            return list(indices+adjustments)
        
    elif vertex_check(point,interval) == 3: #case for inner-vertex
        indices = np.array([get_index(point,interval,ks[2])] * 8,dtype=int)
        adjustments = np.array([0,1,ks[0],ks[0]+1,ks[1],ks[1]+1,+ks[0]+ks[1],ks[0]+ks[1]+1],dtype=int)
        return list(indices+adjustments)
    
    elif sum(edge_check(point,start,end))==2: #case for outermost-edge
        indices = np.array([get_index(point,interval,ks[2])],dtype=int) 
        return list(indices)
    
    elif sum(edge_check(point,start,end))==1: #case for outer-edge
        
        if outer_edge_check(point[1],start,end): #change np.logic
            
            if point[0] in interval:
                indices = np.array([get_index(point,interval,ks[2])]*2,dtype=int)
                adjustments = np.array([0,1])
                return list(indices+adjustments)
            
            else:
                indices = np.array([get_index(point,interval,ks[2])]*2,dtype=int)
                adjustments = np.array([0,ks[1]])
                return list(indices+adjustments) 
            
        elif outer_edge_check(point[0],start,end):
            
            if point[1] in interval:
                indices = np.array([get_index(point,interval,ks[2])]*2,dtype=int)
                adjustments = np.array([0,ks[0]])
                return list(indices+adjustments)
            
            else:
                indices = np.array([get_index(point,interval,ks[2])]*2,dtype=int)
                adjustments = np.array([0,ks[1]])
                return list(indices+adjustments)  
            
        elif outer_edge_check(point[2],start,end):
            
            if point[0] in interval:
                indices = np.array([get_index(point,interval,ks[2])]*2,dtype=int)
                adjustments = np.array([0,1])
                return list(indices+adjustments)
            
            else:
                indices = np.array([get_index(point,interval,ks[2])]*2,dtype=int)
                adjustments = np.array([0,ks[0]])
                return list(indices+adjustments)  
            
    elif vertex_check(point,interval)==2 : #case for inner-edge
        
        if point[0] not in interval:
            indices = np.array([get_index(point,interval,ks[2])]*4,dtype=int)
            adjustments = np.array([0,ks[0],ks[1],ks[0]+ks[1]])
            return list(indices+adjustments)
        
        elif point[1] not in interval:
            indices = np.array([get_index(point,interval,ks[2])]*4,dtype=int)
            adjustments = np.array([0,1,ks[1],ks[1]+1])
            return list(indices+adjustments)
        
        elif point[2] not in interval:
            indices = np.array([get_index(point,interval,ks[2])]*4,dtype=int)
            adjustments = np.array([0,1,ks[0],ks[0]+1])
            return list(indices+adjustments)
        
        
def angle(v1,v2):
    """ This function computes the angle between two vectors. This is used to sort points in order to
    form the polygonal intersection of the plane with the grid. 
    """
    prod = np.dot(v1[0:2],v2[0:2])
    det = v1[0]*v2[1] - v1[1]*v2[0]
    temp_cos = prod/np.linalg.norm(v1[0:2])/np.linalg.norm(v2[0:2])
    
    if temp_cos >=0:
        cos_t = min(1,temp_cos)
        
    else:
        cos_t = max(-1,temp_cos)
        
    theta =  np.arccos(cos_t)
    
    if det <=0:
        return theta
    
    else:
        return theta + np.pi

def point_sorter(points):
    """ Unsorted points are arranged into the polygon the form based on the angles between the 
    vectors. Vectors are drawn from the center of the points.
    """
    cent = sum(points)/len(points)
    vects = cent - points
    projs = [i[0:2] for i in vects]
    sorted_points = []
    angles = []
    
    for i in range(len(projs)):
        angles.append(angle(projs[0], projs[i]))
        
    idx = np.argsort(angles)
    points = np.array(points)
    sorted_points = points[idx]
    sorted_points = sorted_points.tolist()
    return sorted_points

def area_poly(points):
    """ This function computes the area of the polygon formed by the intersection points after
    they are sorted.
    """
    area = 0.5 * abs(points[0][0] * points[-1][1] - points[0][1] * points[-1][0])
    
    for i in range(len(points) - 1):
        area += 0.5 * abs(points[i][0] * points[i + 1][1] - points[i][1] * points[i + 1][0])
    return area

def ransample(n,pi,mu,sigma):
    """ This function draws variables from a multivariate normal distribution. """
    x=np.zeros((n))
    y=np.zeros((n))
    z=np.zeros((n))
    k=np.random.choice(len(pi),n,p=pi,replace=True)
    
    for i in range(0,len(k)):
        x[i],y[i],z[i]=np.random.multivariate_normal(mu[k[i]],sigma[k[i]],1).T
    return x,y,z


def transmatrix_3d(sample,grid):
    """ This function is the sub-function of transmatrix() that is used to generate the transformation matrix
    for the 3-dimensional application of the method. Creates the class object tmatrix which is used as an
    argument to the rmle() function.
    """
    xy_sample = sample.scaled_sample.copy()
    interval=grid.interval
    L=np.zeros((len(xy_sample),grid.numgridpoints()))
    b0s=interval
    b1s=interval
    b2s=interval
    start = grid.start
    end = grid.end
    ks = grid.ks()
    step = grid.step
    
    for n in range(0,len(xy_sample)):
        b0_based_intersections = []
        b1_based_intersections = []
        b2_based_intersections = []
        
        for i in b1s:
            
            for j in b2s:
                b0_int = (xy_sample[n,3] - i*xy_sample[n,1] - j*xy_sample[n,2])/xy_sample[n,0]
                if b0_int >= start and b0_int <= end:
                    b0_based_intersections.append(tuple([b0_int,i,j]))
                    
        for i in b0s:
            
            for j in b2s:
                b1_int = (xy_sample[n,3] - i*xy_sample[n,0] - j*xy_sample[n,2])/xy_sample[n,1]
                if b1_int >= start and b1_int <= end:
                    b1_based_intersections.append(tuple([i,b1_int,j]))
                    
        for i in b0s:
            
            for j in b1s:
                b2_int = (xy_sample[n,3] - i*xy_sample[n,0] - j*xy_sample[n,1])/xy_sample[n,2]
                if b2_int >= start and b2_int <= end:
                    b2_based_intersections.append(tuple([i,j,b2_int]))
                    
        b0_based_intersections.extend(b1_based_intersections)
        b0_based_intersections.extend(b2_based_intersections)
        intersection_points = b0_based_intersections  
        intersect_points = list(set(intersection_points))
        test_list = [[] for i in range(0,ks[2])]
        
        for p in intersect_points:
            indices=get_box_index(np.array(p),start,end,step,interval,ks)
            
            for i in indices:
                test_list[i].append(np.array(p))
                
        areas = [[] for i in range(0,ks[2])]
        i = 0
        
        for ps in test_list:
            if len(ps) > 2:
                areas[i].append(area_poly(point_sorter(ps)))
                i+=1
                
            else:
                areas[i].append(0)
                i+=1
                
        areas_flat = []
        
        for sublist in areas:
            for i in sublist:
                areas_flat.append(i)
        area_indices = list(np.where(np.array(areas_flat)>0)[0])
        
        for i in area_indices:
            L[n][i]=areas_flat[i]
            
    L= L[~np.all(L==0, axis=1)]
    return tmatrix(Tmat=L,grid=grid,scaled_sample=xy_sample,sample=sample.sample)

def second_deriv_3d(f,step):
    """ This function computes the Laplacian of f"""
    f=f+10e-3
    f=f.reshape(int(np.ceil(len(f)**(1/3))),int(np.ceil(len(f)**(1/3))),int(np.ceil(len(f)**(1/3))))
    fgrad0=np.ravel(np.gradient(np.gradient(f,step)[0],step)[0])
    fgrad1=np.ravel(np.gradient(np.gradient(f,step)[1],step)[1])
    fgrad2=np.ravel(np.gradient(np.gradient(f,step)[2],step)[2])
    return fgrad0+fgrad1+fgrad2

def likelihood_3d(f,n,L_mat_long):
    """ This function is the log-likelihood functional without a regularization 
    term.
    """
    L_mat=L_mat_long.reshape(n,len(f))
    f[f<0]=1e-6
    val=np.log(np.dot(L_mat,f))
    return -sum(val)/n

def sobolev_3d(f,alpha,n,L_mat_long,step):
    """ This function computes the value of the functional with the H1 penalty for the 3-d 
    implementation of the method."""
    L_mat=L_mat_long.reshape(n,len(f))
    f[f<0]=0
    f=f+10e-3
    val=np.log(np.dot(L_mat,f))
    return -sum(val)/n + alpha*step**3*sum(f**2)+alpha*step**3*norm_fprime_3d(f,step)

def norm_fprime_3d(f,step):
    """ This function computes the value of the norm of the derivative of f """
    f=f.reshape(int(np.ceil(len(f)**(1/3))),int(np.ceil(len(f)**(1/3))),int(np.ceil(len(f)**(1/3))))
    fgrad=np.gradient(f,step)
    return sum(np.ravel((fgrad[0]**2+fgrad[1]**2+fgrad[2]**2)))

def jac_sobolev_3d(f,alpha,n,L_mat_long,step):
    """ This function computes the value of the jacobian of the functional with the H1 penalty for
    the 3-d implementation of the method.
    """
    f[f<0]=1e-6
    L_mat=L_mat_long.reshape(n,len(f))
    denom=np.dot(L_mat,f)
    val=L_mat.T/denom
    return -val.T.sum(axis=0)/n+alpha*step**3*2*f-2*alpha*step**3*second_deriv_3d(f,step)
 
def norm_sq_3d(f,alpha,n,L_mat_long,step):
    """ This function comptues the value of the functional with the squared L2 norm penalty for the 3-d 
    implementation of the method."""
    L_mat=L_mat_long.reshape(n,len(f))
    f[f<0]=1e-6
    val=np.log(np.dot(L_mat,f))
    return -sum(val)/n+ alpha*step**3*sum(f**2)

def entropy_3d(f,alpha,n,L_mat_long,step):
    """ This function comptues the value of the functional with the entropy penalty for the 3-d 
    implementation of the method."""
    L_mat=L_mat_long.reshape(n,len(f))
    f[f<0]=1e-6
    val=np.log(np.dot(L_mat,f))
    return -sum(val)/n + alpha*step**3*sum(f*np.log(f))


def jac_likelihood_3d(f,n,L_mat_long):
    """ This function computes the jacobian of the regularization
    functional without any penalty term.
    """
    f[f < 0] = 1e-6
    L_mat=L_mat_long.reshape(n,len(f))
    denom=np.dot(L_mat,f)
    val=L_mat.T/denom
    return -val.T.sum(axis=0)/n


def jac_norm_sq_3d(f,alpha,n,L_mat_long,step):
    """ This function computes the value of the jacobian of the functional with the squared L2 norm penalty for
    the 3-d implementation of the method.
    """
    f[f < 0] = 1e-6
    L_mat=L_mat_long.reshape(n,len(f))
    denom=np.dot(L_mat,f)
    val=L_mat.T/denom
    return -val.T.sum(axis=0)/n+alpha*step**3*2*f

def jac_entropy_3d(f,alpha,n,L_mat_long,step):
    """ This function computes the value of the jacobian of the functional with the entropy penalty for
    the 3-d implementation of the method.
    """
    L_mat=L_mat_long.reshape(n,len(f))
    f[f<0]=1e-6
    denom=np.dot(L_mat,f)
    val=L_mat.T/denom
    return -val.T.sum(axis=0)/n+alpha*step**3*(np.log(f)+1)

def likelihood_l(f,L,n):
    L=L.reshape(n,len(f))
    f[f<0]=1e-6
    val=np.log(np.dot(L,f))
    return -sum(val)


def rmle_3d(functional,alpha,tmat,shift=None,k=None,jacobian=None,initial_guess=None,hessian_method=None,constraints=None,tolerance=None,max_iter=None,bounds=None):
    """ This function is a sub-function of rmle(). This is a wrapper function for the SciPy Optimize minimize()
    function. It has 3 essential arguments which are: {functional, alpha, tmat}. 
    
    Essential/Positional arguments:
    
    functional - the choice of regularization functional
    
    alpha - the choice of the alpha parameter which can be a constant or a string matching 'cv' or 'lepskii'
    
    tmat - the class object tmatrix generated by the the transmatrix() function. This contains the 
    transformation matrix used in the evaluation of the functional and its jacobian.
    
    Optional arguments:
    
    shift - When set to true, it implements the shifting algorithm used to prevent the instability in the 
    algorithm when all random coefficients are centralized at zero.
    
    k - an integer value that specifies the number of folds to be used in k-fold cross validation
     
    jacobian - accepts a function as an argument or string type, for more options the please refer to the
    documentation of SciPy's minimize function for more details. In the case of this module, the jacobian of 
    functional's explicit form is provided in the form of a function. We advise to keep this argument to its
    default status.
    
    intiial_guess - an array argument that matches the dimensions of \hat{f_\bbeta}.
     
    hessian_method -  the default is the '2-point' method implemented in the SciPy library. The user can 
    provide the other hessian methods avaiable to the scipy optimize minimize() function.
    
    constraints - the linear constraint for the optimization problem, namely: ||\hat{f_\bbeta}||_L1 = 1 i.e.
    the estimate \hat{f_\bbeta} should integrate to 1.
    
    tolerance - a floating point argument that sets the tolerance value for the optimization problem.
    
    max_iter - an integer that sets the maximum number of iterations.
    
    bounds - sets the bounds for the values of the estimate, the default value is a non-negativity constraint.
    """     
    st = time.time()
    n=tmat.n()
    m=tmat.m()
    trans_matrix_long = np.ravel(tmat.Tmat)
    trans_matrix = tmat.Tmat
    step_size = tmat.grid.step
    sample = tmat.sample.copy()
    lepskii_matches = ['lep','lepskii','lepskii\'s principle', 'lp']
    cv_matches = ['cv','cross','cross val', 'validation','crossvalidation','cross-validation']
    
    if '3d' not in str(functional.__name__):
        raise ValueError("The 3-dimensional implementation requires a matching 3-dimensional functional. \
Try to supply any of the accepted functional values: {likelihood_3d, norm_sq_3d, sobolev_3d, entropy_3d}")
        
    if initial_guess is not None:
        initial_guess = initial_guess
        
    else:
        initial_guess = initial_gauss(tmat.grid)
        
    if not jacobian:
        if functional == sobolev_3d:
            jacobian = jac_sobolev_3d
        elif functional == likelihood_3d:
            jacobian = jac_likelihood_3d
        elif functional == norm_sq_3d:
            jacobian = jac_norm_sq_3d
        elif functional == entropy_3d:
            jacobian = jac_entropy_3d
        else: 
            jacobian = jacobian
            
    if not hessian_method:
        hessian_method = '2-point'
        
    else:
        hessian_method = hessian_method
        
    if not constraints:
        linear_constraint=scop.LinearConstraint([step_size**3]*len(initial_guess),[1],[1])
        constraints = linear_constraint
        
    else:
        constraints = constraints
        
    if not tolerance:
        tolerance = 1e-12
        
    else:
        tolerance = tolerance
        
    if not max_iter:
        max_iter = 10*m
        
    else: 
        max_iter = max_iter
        
    if not bounds:
        bound = scop.Bounds(0,np.inf)
        
    else:
        bound = bounds
        
    if not k:
        k = 10
        
    else:
        k = k
        
    if shift == True:
        num_shifts = 10
        reconstructions = []
        shift_tmatrix_list = []
        i=0
        
        while i < num_shifts:
            shifts = np.array([np.random.uniform(1,2.5)/tmat.grid.b0,0,0]) + np.array(tmat.grid.shifts)
            shifted_grid = grid_obj(interval=tmat.grid.interval,scale=tmat.grid.scale, shifts = shifts, dim=tmat.grid.dim,step=step_size,start=tmat.grid.start,end=tmat.grid.end)
            shift_tmatrix = transmatrix(sample,shifted_grid)
            n=shift_tmatrix.n()
            shift_tmatrix_list.append(shift_tmatrix)
            shift_trans_matrix_long = np.ravel(shift_tmatrix.Tmat)
            rec = scop.minimize(functional,initial_guess,args=(alpha,n,shift_trans_matrix_long,step_size),method='trust-constr',jac=jacobian,hess=hessian_method,constraints=[constraints],tol=tolerance,options={'verbose': 0,'maxiter': max_iter},bounds=bound)
            reconstructions.append(rec.x)
            i+=1
            updt(num_shifts, i)
            
        l2s = []
        for r in reconstructions:
            l2s.append(np.linalg.norm(r))
            
        l2_arr = np.array(l2s).reshape(-1,1)
        kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
        ys = kmeans.fit_predict(l2_arr)
        indices = select_cluster(ys,kmeans,l2s)[0]
        centroid = kmeans.cluster_centers_[select_cluster(ys,kmeans,l2s)[1]][0]
        diff_l2_arr = abs(l2_arr[indices] - centroid)
        min_index = np.where(diff_l2_arr==np.min(diff_l2_arr))[0][0]
        rec_choice = reconstructions[min_index]
        shifted_tmatrix = shift_tmatrix_list[min_index]
        et = time.time()
        details = {
            "Regularization Functional": str(functional.__name__),
            "Jacobian": str(jacobian.__name__),
            "Numer of Grid Points": m,
            "Sample Size": n,
            "Total Run Time": str(et-st) + ' seconds',
            "Number of Shifts Applied": num_shifts,
        }
        return RMLEResult(f=reconstructions[min_index],alpha=alpha,alpmth='User',T=shifted_tmatrix,details=details)
    
    if any(c in str.lower(str(alpha)) for c in lepskii_matches):
        r=1.2
        alphas=alpha_lep(70,n,r)
        reconstructions=[]
        total = len(alphas)
        i_range=np.arange(0,len(alphas))
        j=0
        times=[]
        
        for a in alphas:
            rec = scop.minimize(functional,initial_guess,args=(a,n,trans_matrix_long,step_size),method='trust-constr',jac=jacobian,hess=hessian_method,constraints=[constraints],tol=tolerance,options={'verbose': False,'maxiter': max_iter},bounds=bound)
            reconstructions.append(rec.x)
            j+=1
            updt(total,j)
        approx_errs = []
        
        for i in range(1,len(reconstructions)):
            jj=0
            approx_error = []
            
            while jj < i:
                approx_error.append(np.linalg.norm(reconstructions[i]-reconstructions[jj]))
                jj+=1
            approx_errs.append(approx_error)
        prop_errors=prop_n(r,2,i_range)
        diffs=[]
        
        for p in approx_errs:
            n_p = int(len(p))
            diffs.append(np.array(prop_errors[:n_p])-np.array(p))
        index=index_finder(diffs)
        et = time.time()
        details = {
            "Regularization Functional": str(functional.__name__),
            "Jacobian": str(jacobian.__name__),
            "Numer of Grid Points": m,
            "Sample Size": n,
            "Total Run Time": str(et-st) + ' seconds',
            "Number of Lepskii iterations": len(alphas)
        }
        return RMLEResult(f=reconstructions[index],alpha=alphas[index],alpmth='Lepskii',T=tmat,details=details)
    if any(c in str.lower(str(alpha)) for c in cv_matches):
        alphas=alpha_vals(step_size*3,30)
        lhood=[]
        reconstructions=[]
        alpha_list=[]
        trans_matrix=sample_shuffle(trans_matrix)
        time_start = time.time()
        j=0
        total = np.ceil((np.log(4)-np.log(len(alphas)))/(0.25*np.log(0.25)+0.75*np.log(0.75)))*4 * k
        
        while len(alphas) > 4:
            val=quarter_selector(alphas,n,k,j,time_start,total,functional,initial_guess,trans_matrix,trans_matrix_long,step_size,jacobian,hessian_method,constraints,tolerance,max_iter,bound)
            lhood.append(val[2])
            reconstructions.append(val[0])
            j=val[3]
            alphas=val[1]
            alpha_list.append(val[4])
            
        for a in alphas:
            val=cv_loss(a,alphas,n,k,j,time_start,total,functional,initial_guess,trans_matrix,trans_matrix_long,step_size,jacobian,hessian_method,constraints,tolerance,max_iter,bound)
            lhood.append(val[1])
            reconstructions.append(val[0])
            alpha_list.append(a)
            j=val[2]
        index=np.int(np.where(lhood==np.min(lhood))[0])
        updt_cv(total,total,time_start)
        et = time.time()
        details = {
            "Regularization Functional": str(functional.__name__),
            "Jacobian": str(jacobian.__name__),
            "Numer of Grid Points": m,
            "Sample Size": n,
            "Total Run Time": str(et-st) + ' seconds',
            "Number of CV iterations": j
        }
        return RMLEResult(f=reconstructions[index],alpha=alpha_list[index],alpmth='CV',T=tmat,details=details)
    
    else:
        result = scop.minimize(functional,initial_guess,args=(alpha,n,trans_matrix_long,step_size),method='trust-constr',jac=jacobian,hess=hessian_method,constraints=[constraints],tol=tolerance,options={'verbose': 0,'maxiter': max_iter},bounds=bound)
        et = time.time()
        details = {
            "Regularization Functional": str(functional.__name__),
            "Jacobian": str(jacobian.__name__),
            "Numer of Grid Points": m,
            "Sample Size": n,
            "Total Run Time": str(et-st) + ' seconds',
            "Iterations": result.niter,
            "Optimization Run Time": result.execution_time
        }
        return RMLEResult(f=result.x,alpha=alpha,alpmth='User',T=tmat,details=details)
    
def scale_sample(xy_sample,grid):
    """ This function is used to apply the shifts and scaling to the sample. """
    sample = xy_sample.copy()
    
    if grid.dim == 2:
        sample[:,2] = sample[:,2] + grid.shifts[0]*sample[:,0] + grid.shifts[1]*sample[:,1]
        sample[:,0] = sample[:,0]/grid.b0
        sample[:,1] = sample[:,1]/grid.b1 
        return sample
    
    else:
        sample[:,3] = sample[:,3] + grid.shifts[0]*sample[:,0] + grid.shifts[1]*sample[:,1] + grid.shifts[2]*sample[:,2]
        sample[:,0] = sample[:,0]/grid.b0
        sample[:,1] = sample[:,1]/grid.b1
        sample[:,2] = sample[:,2]/grid.b2
        return sample
    
class shft_sample_obj:
    """ Class used to store the sample and the shifts applied to it Output of the function shift_sample.""" 
    def __init__(self,sample,shifts):
        self.sample = sample
        self.shifts = shifts

class sample_obj:
    """ Class used to store the sample and the scaled sample."""
    def __init__(self,scaled_sample,sample):
        self.scaled_sample = scaled_sample
        self.sample = sample
        
def shift_sample(sample,base_shifts):
    """ This function is used to apply shifts to the sample. """
    shift_sample_obs = sample.copy()
    n = np.shape(shift_sample_obs)[1]
    shifts = [np.random.uniform(1,2.5) for i in range(0,n-1)] + np.array(base_shifts)
    
    for i in range(0,n-1):
        shift_sample_obs[:,n-1]=shifts[i]*shift_sample_obs[:,i]+shift_sample_obs[:,n-1]
    return shft_sample_obj(shift_sample_obs,shifts)
