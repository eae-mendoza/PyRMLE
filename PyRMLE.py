from PyRMLE_hfuncs import *
    
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


def plot_rmle(result, plt_type=None, save_fig=None):
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
    new_interval = np.array(
        [(result.grid.interval[i - 1] + result.grid.interval[i]) / 2 for i in range(1, len(result.grid.interval))])
    b0_scale = result.grid.b0
    b1_scale = result.grid.b1
    b2_scale = result.grid.b2
    shifts = result.grid.shifts
    if not plt_type:
        if dim == 2:
            shaped = result.f.reshape(m, m)
            B0 = (new_interval) / b0_scale - shifts[0]
            B1 = (new_interval) / b1_scale - shifts[1]
            contour = plt.contour(B0, B1, shaped, colors='black')
            plt.clabel(contour, inline=True, fontsize=8)
            plt.imshow(shaped, extent=[min(B0), max(B0), min(B1), max(B1)], cmap='OrRd', alpha=0.5)
            plt.colorbar()
            if save_fig is not None:
                plt.savefig('{filename}.png'.format(filename=save_fig))
            plt.show()
        else:
            shaped = result.f.reshape(m, m, m)
            f12 = np.sum(shaped, axis=2) * step_size
            f01 = np.sum(shaped, axis=0) * step_size
            f02 = np.sum(shaped, axis=1) * step_size
            B0 = (new_interval) / b0_scale - shifts[0]
            B1 = (new_interval) / b1_scale - shifts[1]
            B2 = (new_interval) / b2_scale - shifts[2]
            contour1 = plt.contour(B0, B1, f01, colors='black')
            plt.clabel(contour1, inline=True, fontsize=8)
            plt.imshow(f01, extent=[min(B0), max(B0), min(B1), max(B1)], origin='lower', cmap='OrRd', alpha=0.5)
            plt.colorbar()
            if save_fig is not None:
                plt.savefig('{filename}_f01.png'.format(filename=save_fig))
            plt.show()
            contour2 = plt.contour(B0, B2, f02, colors='black')
            plt.clabel(contour2, inline=True, fontsize=8)
            plt.imshow(f02, extent=[min(B0), max(B0), min(B2), max(B2)], origin='lower', cmap='OrRd', alpha=0.5)
            plt.colorbar()
            if save_fig is not None:
                plt.savefig('{filename}_f02.png'.format(filename=save_fig))
            plt.show()
            contour3 = plt.contour(B1, B2, f12, colors='black')
            plt.clabel(contour3, inline=True, fontsize=8)
            plt.imshow(f12, extent=[min(B1), max(B1), min(B2), max(B2)], origin='lower', cmap='OrRd', alpha=0.5)
            plt.colorbar()
            if save_fig is not None:
                plt.savefig('{filename}_f12.png'.format(filename=save_fig))
            plt.show()
    elif 'surf' in str(plt_type):
        if dim == 2:
            shaped = result.f.reshape(m, m)
            B0 = (new_interval) / b0_scale - shifts[0]
            B1 = (new_interval) / b1_scale - shifts[1]
            b0_axis, b1_axis = np.meshgrid(B0, B1)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(b0_axis, b1_axis, shaped, cmap='OrRd', linewidth=0, alpha=1)
            ax.set_xlabel('B0')
            ax.set_ylabel('B1')
            ax.set_zlabel('f_B  ')
            ax.view_init(elev=30, azim=100)
            fig.colorbar(plot, ax=ax)
            if save_fig is not None:
                plt.savefig('{filename}.png'.format(filename=save_fig))
            plt.show()
        else:
            shaped = result.f.reshape(m, m, m)
            f12 = np.sum(shaped, axis=2) * step_size
            f01 = np.sum(shaped, axis=0) * step_size
            f02 = np.sum(shaped, axis=1) * step_size
            B0 = (new_interval) / b0_scale - shifts[0]
            B1 = (new_interval) / b1_scale - shifts[1]
            B2 = (new_interval) / b2_scale - shifts[2]
            b0_axis, b1_axis = np.meshgrid(B0, B1)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            plot = ax.plot_surface(b0_axis, b1_axis, f01, cmap='OrRd', linewidth=0, alpha=1)
            ax.set_xlabel('B0')
            ax.set_ylabel('B1')
            ax.set_zlabel('f_B  ')
            fig.colorbar(plot, ax=ax)
            ax.view_init(elev=30, azim=100)
            if save_fig is not None:
                plt.savefig('{filename}_f01.png'.format(filename=save_fig))
            plt.show()
            b0_axis, b2_axis = np.meshgrid(B0, B2)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            plot = ax.plot_surface(b0_axis, b2_axis, f02, cmap='OrRd', linewidth=0, alpha=1)
            ax.set_xlabel('B0')
            ax.set_ylabel('B2')
            ax.set_zlabel('f_B  ')
            ax.view_init(elev=30, azim=100)
            fig.colorbar(plot, ax=ax)
            if save_fig is not None:
                plt.savefig('{filename}_f02.png'.format(filename=save_fig))
            plt.show()
            b1_axis, b2_axis = np.meshgrid(B1, B2)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            plot = ax.plot_surface(b1_axis, b2_axis, f12, cmap='OrRd', linewidth=0, alpha=1)
            ax.set_xlabel('B1')
            ax.set_ylabel('B2')
            ax.set_zlabel('f_B  ')
            fig.colorbar(plot, ax=ax)
            ax.view_init(elev=30, azim=100)
            if save_fig is not None:
                plt.savefig('{filename}_f12.png'.format(filename=save_fig))
            plt.show()
