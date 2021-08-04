from pyrmle_hfuncs import *

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
        return sim_sample2d(n,x_params,beta_pi,beta_mu,beta_cov)
    
    else:
        return sim_sample3d(n,x_params,beta_pi,beta_mu,beta_cov)

def spline_fit(result,num_grid_points):
    dim = result.dim
    
    if dim == 2:
        spline_x = result.grid.b0_grid_points
        spline_y = result.grid.b1_grid_points
        spline_z = result.f_shaped
        f = sp.interpolate.interp2d(spline_x, spline_y, spline_z, kind='cubic')
        x_grid = np.linspace(spline_x[0], spline_x[-1], num_grid_points)
        y_grid = np.linspace(spline_y[0], spline_y[-1], num_grid_points)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        spline_znew = f(x_grid, y_grid)
        spline_grid = spline_grid_obj(num_grid_points = num_grid_points, b0_grid_points=x_grid,b1_grid_points=y_grid,b2_grid_points=None,dim=dim,scale=result.grid.scale,shifts=result.grid.shifts)
        
        return SplineResult(f= np.ravel(spline_znew), f_shaped= spline_znew, joint_marginals= None, spline_grid= spline_grid)
    
    elif dim == 3:
        # 3-D array interpolation
        spline_x = result.grid.b0_grid_points
        spline_y = result.grid.b1_grid_points
        spline_z = result.grid.b2_grid_points
        
        gg = result.f_shaped
        
        interp_func = sp.interpolate.RegularGridInterpolator((spline_x, spline_y, spline_z), gg)
        
        x_grid = np.linspace(spline_x[0], spline_x[-1], num_grid_points)
        y_grid = np.linspace(spline_y[0], spline_y[-1], num_grid_points)
        z_grid = np.linspace(spline_z[0], spline_z[-1], num_grid_points)
        
        xg, yg, zg = np.meshgrid(x_grid, y_grid, z_grid)
        interp_znew = interp_func((xg, yg, zg))
        # Determine the marginals via smooth spline interpolation
        
        f12 = np.sum(result.f_shaped, axis=2) * (result.grid.b0_grid_points[1] - result.grid.b0_grid_points[0])
        f01 = np.sum(result.f_shaped, axis=0) * (result.grid.b1_grid_points[1] - result.grid.b1_grid_points[0])
        f02 = np.sum(result.f_shaped, axis=1) * (result.grid.b2_grid_points[1] - result.grid.b2_grid_points[0])
        
        spline_f01_func = sp.interpolate.interp2d(spline_x, spline_y, f01, kind='cubic')
        spline_f02_func = sp.interpolate.interp2d(spline_x, spline_z, f01, kind='cubic')
        spline_f12_func = sp.interpolate.interp2d(spline_y, spline_z, f01, kind='cubic')
        
        f01_spline = spline_f01_func(x_grid,y_grid)
        f02_spline = spline_f02_func(x_grid,z_grid)
        f12_spline = spline_f12_func(y_grid,z_grid)
        jmarginals = [f01_spline,f02_spline,f12_spline]
        spline_grid = spline_grid_obj(num_grid_points = num_grid_points, b0_grid_points=x_grid,b1_grid_points=y_grid,b2_grid_points=z_grid, dim=dim, scale=result.grid.scale,shifts=result.grid.shifts)
        
        return SplineResult(f=np.ravel(interp_znew),f_shaped=interp_znew,joint_marginals=  jmarginals, spline_grid = spline_grid)


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
    plt_matches = ['surf','sf','surface','3d']
    
    if type(result) == RMLEResult:
        m = result.grid.ks()[0]
        step_size = result.grid.step
        
        if not plt_type:
            if dim == 2:
                shaped = result.f_shaped
                B0 = result.grid.b0_grid_points
                B1 = result.grid.b1_grid_points
                contour = plt.contour(B0, B1, shaped, colors='black')
                plt.clabel(contour, inline=True, fontsize=8)
                plt.imshow(shaped, extent=[min(B0), max(B0), min(B1), max(B1)], origin = 'lower',cmap='OrRd', alpha=0.5)
                plt.colorbar()
                if save_fig is not None:
                    plt.savefig('{filename}.png'.format(filename=save_fig))
                plt.show()
                
            else:
                shaped = result.f_shaped
                f12 = np.sum(shaped, axis=2) * step_size
                f01 = np.sum(shaped, axis=0) * step_size
                f02 = np.sum(shaped, axis=1) * step_size
                B0 = result.grid.b0_grid_points
                B1 = result.grid.b1_grid_points
                B2 = result.grid.b2_grid_points
                plt.figure(1)
                contour1 = plt.contour(B0, B1, f01, colors='black')
                plt.clabel(contour1, inline=True, fontsize=8)
                plt.imshow(f01, extent=[min(B0), max(B0), min(B1), max(B1)], origin='lower', cmap='OrRd', alpha=0.5)
                plt.colorbar()
                if save_fig is not None:
                    plt.savefig('{filename}_f01.png'.format(filename=save_fig))
                plt.show()
                
                plt.figure(2)
                contour2 = plt.contour(B0, B2, f02, colors='black')
                plt.clabel(contour2, inline=True, fontsize=8)
                plt.imshow(f02, extent=[min(B0), max(B0), min(B2), max(B2)], origin='lower', cmap='OrRd', alpha=0.5)
                plt.colorbar()
                if save_fig is not None:
                    plt.savefig('{filename}_f02.png'.format(filename=save_fig))
                plt.show()
                
                plt.figure(3)
                contour3 = plt.contour(B1, B2, f12, colors='black')
                plt.clabel(contour3, inline=True, fontsize=8)
                plt.imshow(f12, extent=[min(B1), max(B1), min(B2), max(B2)], origin='lower', cmap='OrRd', alpha=0.5)
                plt.colorbar()
                if save_fig is not None:
                    plt.savefig('{filename}_f12.png'.format(filename=save_fig))
                plt.show()
                
        elif any(c in str.lower(str(plt_type)) for c in plt_matches):
            if dim == 2:
                shaped = result.f_shaped
                B0 = result.grid.b0_grid_points
                B1 = result.grid.b1_grid_points
                b0_axis, b1_axis = np.meshgrid(B0, B1)
                fig = plt.figure(1)
                ax = fig.add_subplot(projection='3d')
                plot = ax.plot_surface(b0_axis, b1_axis, shaped, cmap='OrRd', linewidth=0, alpha=1)
                ax.set_xlabel('B0')
                ax.set_ylabel('B1')
                ax.set_zlabel('f_B  ')
                ax.view_init(elev=30, azim=100)
                fig.colorbar(plot, ax=ax)
                if save_fig is not None:
                    plt.savefig('{filename}.png'.format(filename=save_fig))
                plt.show()
                
            else:
                shaped = result.f_shaped
                f12 = np.sum(shaped, axis=2) * step_size
                f01 = np.sum(shaped, axis=0) * step_size
                f02 = np.sum(shaped, axis=1) * step_size
                B0 = result.grid.b0_grid_points
                B1 = result.grid.b1_grid_points
                B2 = result.grid.b2_grid_points
                b0_axis, b1_axis = np.meshgrid(B0, B1)
                fig = plt.figure(1)
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
                fig = plt.figure(2)
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
                fig = plt.figure(3)
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
                
    elif type(result) ==  SplineResult:
        if not plt_type:
            
            if dim == 2:
                shaped = result.f_shaped
                B0 = result.grid.b0_grid_points
                B1 = result.grid.b1_grid_points
                contour = plt.contour(B0, B1, shaped, colors='black')
                plt.clabel(contour, inline=True, fontsize=8)
                plt.imshow(shaped, extent=[min(B0), max(B0), min(B1), max(B1)], origin = 'lower',cmap='OrRd', alpha=0.5)
                plt.colorbar()
                if save_fig is not None:
                    plt.savefig('{filename}.png'.format(filename=save_fig))
                plt.show()
                
            else:
                f12 = result.joint_marginals[2]
                f01 = result.joint_marginals[0]
                f02 = result.joint_marginals[0]
                B0 = result.grid.b0_grid_points
                B1 = result.grid.b1_grid_points
                B2 = result.grid.b2_grid_points
                plt.figure(1)
                contour1 = plt.contour(B0, B1, f01, colors='black')
                plt.clabel(contour1, inline=True, fontsize=8)
                plt.imshow(f01, extent=[min(B0), max(B0), min(B1), max(B1)], origin='lower', cmap='OrRd', alpha=0.5)
                plt.colorbar()
                if save_fig is not None:
                    plt.savefig('{filename}_f01.png'.format(filename=save_fig))
                plt.show()
                
                plt.figure(2)
                contour2 = plt.contour(B0, B2, f02, colors='black')
                plt.clabel(contour2, inline=True, fontsize=8)
                plt.imshow(f02, extent=[min(B0), max(B0), min(B2), max(B2)], origin='lower', cmap='OrRd', alpha=0.5)
                plt.colorbar()
                if save_fig is not None:
                    plt.savefig('{filename}_f02.png'.format(filename=save_fig))
                plt.show()
                
                plt.figure(3)
                contour3 = plt.contour(B1, B2, f12, colors='black')
                plt.clabel(contour3, inline=True, fontsize=8)
                plt.imshow(f12, extent=[min(B1), max(B1), min(B2), max(B2)], origin='lower', cmap='OrRd', alpha=0.5)
                plt.colorbar()
                if save_fig is not None:
                    plt.savefig('{filename}_f12.png'.format(filename=save_fig))
                plt.show()
                
        elif any(c in str.lower(str(plt_type)) for c in plt_matches):
            if dim == 2:
                shaped = result.f_shaped
                B0 = result.grid.b0_grid_points
                B1 = result.grid.b1_grid_points
                b0_axis, b1_axis = np.meshgrid(B0, B1)
                fig = plt.figure(1)
                ax = fig.add_subplot(projection='3d')
                plot = ax.plot_surface(b0_axis, b1_axis, shaped, cmap='OrRd', linewidth=0, alpha=1)
                ax.set_xlabel('B0')
                ax.set_ylabel('B1')
                ax.set_zlabel('f_B  ')
                ax.view_init(elev=30, azim=100)
                fig.colorbar(plot, ax=ax)
                if save_fig is not None:
                    plt.savefig('{filename}.png'.format(filename=save_fig))
                plt.show()
                
            else:
                f12 = result.joint_marginals[2]
                f01 = result.joint_marginals[0]
                f02 = result.joint_marginals[0]
                B0 = result.grid.b0_grid_points
                B1 = result.grid.b1_grid_points
                B2 = result.grid.b2_grid_points
                b0_axis, b1_axis = np.meshgrid(B0, B1)
                fig = plt.figure(1)
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
                fig = plt.figure(2)
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
                fig = plt.figure(3)
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
