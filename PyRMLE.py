from PyRMLE_hfuncs import *

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
        B2 = self.grid.b2_grid_pointsgr
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
        B0 = self.grid.b0_grid_points
        B1 = self.grid.b1_grid_points
        B2 = self.grid.b2_grid_points
        # Initialize the list for expected values
        expected_vals = []
        m = self.grid.ks()[0]
        # Call \hat{f_\beta}
        f = self.f
        # Declare the step size of the grid based on the grid used to create the transformation matrix
        b0_step = self.grid.step/self.grid.b0
        b1_step = self.grid.step/self.grid.b1
        b2_step = self.grid.step/self.grid.b2
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
            for i in range(0,grid_est.ks()[0]):
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
    
def grid_set(num_grid_points,dim,B0_range=None,B1_range=None,B2_range=None):
    # Check if custom grid ranges are declared by the user, else use the default 
    if B0_range is not None:
        B0_range = B0_range
    else:
        B0_range = [-5,5]
    if B1_range is not None:
        B1_range = B1_range
    else:
        B1_range = [-5,5]
    if B2_range is not None:
        B2_range = B2_range
    else:
        B2_range = [-5,5]
    # Compute the shift in the center of the grid for each axis, which is simply the mid-point of
    # the new ranges
    shifts = -1*np.array([np.mean(B0_range),np.mean(B1_range),np.mean(B2_range)])
    # Compute for the step size (with respect to the base grid [-5,5])
    step = 10/num_grid_points
    temp_interval = np.arange(-5,5+step,step)
    # We truncate the points by rounding-off since repeating decimals are problematic
    interval = [round(i,6) for i in temp_interval]
    # Set the start and the end of the grid
    start = interval[0]
    end = interval[-1]
    base_scale = interval[-1]-interval[0]
    # Compute the amount by which the grid has to be scaled
    B0_scale_val = base_scale/(B0_range[-1]-B0_range[0])
    B1_scale_val = base_scale/(B1_range[-1]-B1_range[0])
    B2_scale_val = base_scale/(B2_range[-1]-B2_range[0])
    scales = np.array([B0_scale_val,B1_scale_val,B2_scale_val])
    return grid_obj(interval=interval,scale=scales, shifts = shifts, dim=dim,step=step,start=start,end=end)

def rmle(functional,alpha,tmat,shift=None,k=None,jacobian=None,initial_guess=None,hessian_method=None,constraints=None,tolerance=None,max_iter=None,bounds=None):
    """ This function serves as a wrapper function for rmle_2d() and rmle_3d. """
    if tmat.grid.dim == 2:
        result = rmle_2d(functional,alpha,tmat,shift,k,jacobian,initial_guess,hessian_method,constraints,tolerance,max_iter,bounds)
        return result
    else:
        result = rmle_3d(functional,alpha,tmat,shift,k,jacobian,initial_guess,hessian_method,constraints,tolerance,max_iter,bounds)
        return result

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

def sim_sample(n,d,x_params=None,beta_pi=None,beta_mu=None,beta_cov=None):
    if d == 2:
        return sim_sample2d(n,x_params=None,beta_pi=None,beta_mu=None,beta_cov=None)
    else:
        return sim_sample3d(n,x_params=None,beta_pi=None,beta_mu=None,beta_cov=None)

def plot_rmle(result,plt_type=None,save_fig=None):
    """ This function is used to plot the output of the rmle() function.
    Essential argument:
    result - class object RMLEResult. The function extracts all the necessary data in order to plot
    the resulting estimate for \hat{f_\bbeta}.
    
    Optional arguments:
    
    plt_type - default returns contour plots, accepts a string that matches for 'surf' when provided
    would plot a surface plot of \hat{f_\bbeta} or its marginal distributions.
    
    save_fig - a string that serves as the file name of the figure.
    """
    dim = result.dim
    m = result.grid.ks()[0]
    step_size = result.grid.step
    new_interval = np.array([(result.grid.interval[i-1]+result.grid.interval[i])/2 for i in range(1,len(result.grid.interval))])
    b0_scale = result.grid.b0
    b1_scale = result.grid.b1
    b2_scale = result.grid.b2
    shifts = result.grid.shifts
    if not plt_type:
        if dim == 2:
            shaped = result.f.reshape(m,m)
            B0=(new_interval)/b0_scale-shifts[0]
            B1=(new_interval)/b1_scale-shifts[1]
            contour = plt.contour(B0,B1,shaped,colors='black')
            plt.clabel(contour,inline=True,fontsize=8)
            plt.imshow(shaped, extent=[min(B0),max(B0), min(B1), max(B1)],cmap='OrRd', alpha=0.5)
            plt.colorbar()
            if save_fig is not None:
                plt.savefig('{filename}.png'.format(filename=save_fig))
            plt.show()
        else:
            shaped = result.f.reshape(m,m,m)
            f12=np.sum(shaped,axis=2)*step_size
            f01=np.sum(shaped,axis=0)*step_size
            f02=np.sum(shaped,axis=1)*step_size
            B0=(new_interval)/b0_scale-shifts[0]
            B1=(new_interval)/b1_scale-shifts[1]
            B2=(new_interval)/b2_scale-shifts[2]
            contour1 = plt.contour(B0,B1,f01,colors='black')
            plt.clabel(contour1,inline=True,fontsize=8)
            plt.imshow(f01, extent=[min(B0),max(B0), max(B1), min(B1)],cmap='OrRd', alpha=0.5)
            plt.colorbar()
            if save_fig is not None:
                plt.savefig('{filename}_f01.png'.format(filename=save_fig))            
            plt.show()
            contour2 = plt.contour(B0,B2,f02,colors='black')
            plt.clabel(contour2,inline=True,fontsize=8)
            plt.imshow(f02,extent=[min(B0),max(B0), max(B2), min(B2)],cmap='OrRd', alpha=0.5)
            plt.colorbar()
            if save_fig is not None:
                plt.savefig('{filename}_f02.png'.format(filename=save_fig))
            plt.show()
            contour3 = plt.contour(B1,B2,f12,colors='black')
            plt.clabel(contour3,inline=True,fontsize=8)
            plt.imshow(f12, extent=[min(B1),max(B1), max(B2), min(B2)],cmap='OrRd', alpha=0.5)
            plt.colorbar()
            if save_fig is not None:
                plt.savefig('{filename}_f12.png'.format(filename=save_fig))
            plt.show()
    elif 'surf' in str(plt_type):
        if dim == 2:
            shaped = result.f.reshape(m,m)
            B0=(new_interval)/b0_scale-shifts[0]
            B1=(new_interval)/b1_scale-shifts[1]
            b0_axis, b1_axis = np.meshgrid(B0,B1)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(b0_axis, b1_axis, shaped,cmap='OrRd',linewidth=0,alpha=1)
            ax.set_xlabel('B0')
            ax.set_ylabel('B1')
            ax.set_zlabel('f_B  ')
            ax.view_init(elev=30, azim=100)
            fig.colorbar(plot,ax=ax)
            if save_fig is not None:
                plt.savefig('{filename}.png'.format(filename=save_fig))
            plt.show()
        else:
            shaped = result.f.reshape(m,m,m)
            f12=np.sum(shaped,axis=2)*step_size
            f01=np.sum(shaped,axis=0)*step_size
            f02=np.sum(shaped,axis=1)*step_size
            B0=(new_interval)/b0_scale-shifts[0]
            B1=(new_interval)/b1_scale-shifts[1]
            B2=(new_interval)/b2_scale-shifts[2]
            b0_axis, b1_axis = np.meshgrid(B0,B1)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            plot = ax.plot_surface(b0_axis, b1_axis, f01,cmap='OrRd',linewidth=0,alpha=1)
            ax.set_xlabel('B0')
            ax.set_ylabel('B1')
            ax.set_zlabel('f_B  ')
            fig.colorbar(plot,ax=ax)
            ax.view_init(elev=30, azim=100)
            if save_fig is not None:
                plt.savefig('{filename}_f01.png'.format(filename=save_fig))
            plt.show()
            b0_axis, b2_axis = np.meshgrid(B0,B2)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            plot = ax.plot_surface(b0_axis, b2_axis, f02,cmap='OrRd',linewidth=0,alpha=1)
            ax.set_xlabel('B0')
            ax.set_ylabel('B2')
            ax.set_zlabel('f_B  ')
            ax.view_init(elev=30, azim=100)
            fig.colorbar(plot,ax=ax)
            if save_fig is not None:
                plt.savefig('{filename}_f02.png'.format(filename=save_fig))
            plt.show()
            b1_axis, b2_axis = np.meshgrid(B1,B2)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            plot = ax.plot_surface(b1_axis, b2_axis, f12,cmap='OrRd',linewidth=0,alpha=1)
            ax.set_xlabel('B1')
            ax.set_ylabel('B2')
            ax.set_zlabel('f_B  ')
            fig.colorbar(plot,ax=ax)
            ax.view_init(elev=30, azim=100)
            if save_fig is not None:
                plt.savefig('{filename}_f12.png'.format(filename=save_fig))
            plt.show()
