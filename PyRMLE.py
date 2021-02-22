# 1-D functions
#loading in libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.special import gamma
import random
from scipy.integrate import quad,dblquad,tplquad
import scipy.integrate as integrate
import time
from sklearn import preprocessing
import sympy
import sympy as sy
from scipy.signal import *
import scipy.optimize as scop
from multiprocessing import Pool
import math
import sklearn
import multiprocessing as mp
import random
import string
from timeit import default_timer as timer
from matplotlib.lines import Line2D

def ransample_bivar(n,pi,mu,sigma):
    x=np.zeros((n))
    y=np.zeros((n))
    k=np.random.choice(len(pi),n,p=pi,replace=True)
    for i in range(0,len(k)):
        x[i],y[i]=np.random.multivariate_normal(mu[k[i]],cov[k[i]],1).T
    return x,y

def dt_mtx(a,b,discretezation):
    x1=np.arange(a,b,discretezation)
    x2=np.arange(a,b,discretezation)
    coord_mtx=np.zeros((len(x1),len(x2),2))
    for i in range(0,len(x1)):
        coord_mtx[:,i]=np.array((x1,np.full(len(x1),x2[i]))).T
    return coord_mtx

def sim_sample1d(n,x_params=None,beta_params=None):
	if not x_params:
		x_params = [0,1]
	else:
		x_params = x_params
	if not beta_params:
		beta_params = [[0.5,0.5],[[1.5,1.5],[-1.5,-1.5]], [[[1, 0], [0, 1]],[[1, 0], [0, 1]]]]
	else:
		beta_params = beta_params
	x_1=np.random.normal(x_params[0],x_params[1],n).T
	x_0=np.repeat(1,n)
	t={'col1':x_0, 'col2': x_1}
	test=np.array(pd.DataFrame(t))
	x_sample=test
	b0,b1=ransample_bivar(n,beta_params[0],beta_params[1],beta_params[2])
	b={'col1':b0,'col2':b1}
	beta_test=np.array(pd.DataFrame(b))
	#getting the array B*X
	bx=beta_test*test
	#Creating Y
	y=np.array([sum(x) for x in bx])
	xy_sample=np.c_[test,y]
	return xy_sample

def filt(start,end,step,array):
    t=array[:,0]>=start
    array=array[t]
    t1=array[:,0]<=end-step
    array=array[t1]
    t2=array[:,1]>=start
    array=array[t2]
    t3=array[:,1]<=end-step
    array=array[t3]
    return array

def dist_xy(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def filt2(array,slope):
    n=len(array)
    if (slope[1]<=0 and slope[0]>=0) or (slope[1]>=0 and slope[0]<=0):
        return array[1:n]
    else:
        return array[1:n]

def box_avg(grid,i,j):
    avg=grid[i][j]+grid[i+1][j]+grid[i][j+1]+grid[i+1][j+1]
    return avg/4

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

def likelihood_wrapper(f0,sample,b1s,b0s,start,end):
    fs=np.array([f_yx_test(f0,i[0],i[1],b1s,b0s,start,end) for i in sample])
    return -sum(np.log(fs))


def big_L(sample_size,k):
    coord_mtx=np.zeros((sample_size,k))
    return coord_mtx

def which_ij(point,slope):
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
    val=np.array([x[1]*np.sqrt(k)+x[0] for x in ijs])
    return val

def get_intervals(xi,yi):
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
    intervals=[np.linalg.norm(new_b1b0[i]-new_b1b0[i+1]) for i in range(0,len(new_b1b0)-1)]
    reduced_b1b0=filt2(new_b1b0,xi)
    return intervals

def get_intersections(xi,yi):
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
    intervals=[np.linalg.norm(new_b1b0[i]-new_b1b0[i+1]) for i in range(0,len(new_b1b0)-1)]
    reduced_b1b0=filt2(new_b1b0,xi)
    return reduced_b1b0

def create_L(sample,f_dimension):
    L=big_L(len(sample),f_dimension)
    for n in range(0,len(sample)):
        intervals=get_intervals(sample[n],sample[n][2])
        intersection=get_intersections(sample[n],sample[n][2])
        indices=index_conv([which_ij(p,sample[n]) for p in intersection],f_dimension)
        indices=list(map(int,indices))
        for i in range(0,len(indices)):
            L[n][indices[i]]=intervals[i]
    return L

def likelihood_l(f,L):
    f[f<0]=10**-16
    val=np.log(np.dot(L,f))
    return -sum(val)
    
def no_penal(f,n,L_mat_long):
	import numpy as np
	L_mat=L_mat_long.reshape(n,len(f))
	f[f<0]=10**-6
	val=np.log(np.dot(L_mat,f))
	return -sum(val)/n


def norm2_penal(f,alpha,n,L_mat_long,step):
	import numpy as np
	L_mat=L_mat_long.reshape(n,len(f))
	f[f<0]=10**-6
	val=np.log(np.dot(L_mat,f))
	return -sum(val)/n+ alpha*step**2*sum(f**2)

def sobolev_norm_penal(f,alpha,n,L_mat_long,step):
    import numpy as np
    L_mat=L_mat_long.reshape(n,len(f))
    f[f<0]=10**-6
    val=np.log(np.dot(L_mat,f))
    return -sum(val)/n + alpha*step**2*sum(f**2)+alpha*step**2*norm_fprime(f,step)

def entropy_penal(f,alpha,n,L_mat_long,step):
	import numpy as np
	L_mat=L_mat_long.reshape(n,len(f))
	f[f<0]=10**-6
	val=np.log(np.dot(L_mat,f))
	return -sum(val)/n + alpha*step**2*sum(f*np.log(f))


def likelihood_hess(f,L):
	import numpy as np
	dldf2=[None]*len(f)
	denom=np.dot(L,f)
	for i in range(0,len(f)):
		e=np.zeros(len(f))
		e[i]=1
		dldf2[i]=sum(np.dot(L,e)/denom**2)
	return np.array(dldf2)

def tot_deriv(f,step):
	import numpy as np
	f=f.reshape(int(np.sqrt(len(f))),int(np.sqrt(len(f))))
	fgrad=np.gradient(f,step)
	return np.ravel((np.sqrt(fgrad[0]**2+fgrad[1]**2)))

def norm_fprime(f,step):
	import numpy as np
	f=f.reshape(int(np.sqrt(len(f))),int(np.sqrt(len(f))))
	fgrad=np.gradient(f,step)
	return sum(np.ravel((fgrad[0]**2+fgrad[1]**2)))

def second_deriv(f,step):
	import numpy as np
	f=f.reshape(int(np.sqrt(len(f))),int(np.sqrt(len(f))))
	fgrad0=np.ravel(np.gradient(np.gradient(f,step)[0],step)[0])
	fgrad1=np.ravel(np.gradient(np.gradient(f,step)[1],step)[1])
	return fgrad0+fgrad1

def jac_no_penal(f,n,L_mat_long):
	import numpy as np
	L_mat=L_mat_long.reshape(n,len(f))
	denom=np.dot(L_mat,f)
	val=L_mat.T/denom
	return -val.T.sum(axis=0)/n


def jac_norm2_penal(f,alpha,n,L_mat_long,step):
	import numpy as np
	L_mat=L_mat_long.reshape(n,len(f))
	denom=np.dot(L_mat,f)
	val=L_mat.T/denom
	return -val.T.sum(axis=0)/n+alpha*step**2*2*f

"""def jac_sobolev2(f,alpha,n):
    denom=np.dot(L_mat,f)
    val=L_mat.T/denom
    return -val.T.sum(axis=0)/n+alpha*step**2*2*f+2*alpha*step**2*second_deriv(f)"""

def jac_sobolev_norm_penal(f,alpha,n,L_mat_long,step):
	import numpy as np
	L_mat=L_mat_long.reshape(n,len(f))
	denom=np.dot(L_mat,f)
	val=L_mat.T/denom
	return -val.T.sum(axis=0)/n-2*alpha*step**2*second_deriv(f,step)

def jac_entropy_penal(f,alpha,n,L_mat_long,step):
	import numpy as np
	L_mat=L_mat_long.reshape(n,len(f))
	denom=np.dot(L_mat,f)
	val=L_mat.T/denom
	return -val.T.sum(axis=0)/n+alpha*step**2*(np.log(f)+1)

def ise(f1,f2):
    val=np.linalg.norm(f1-f2)**2
    return len(f1)**-1*step**2*val

def prop_noise(alpha,n,C):
    val=C*alpha**-1*n**-1.5
    return val

def get_int(x):
    if any(x<0):
        return next(i for i in x if i<=0)
    else:
        return x[-1]

def rmle_1d(functional,alpha,trans_matrix,step_size,jacobian=None,initial_guess=None,hessian_method=None,constraints=None,tolerance=None,max_iter=None,bounds=None):
    n=len(trans_matrix)
    m=len(np.transpose(trans_matrix))
    trans_matrix_long = np.ravel(trans_matrix)
    if not initial_guess:
        initial_guess = np.zeros(m)+0.000001
    else:
        initial_guess = initial_guess
    if not jacobian:
        if functional == no_penal:
            jacobian = jac_no_penal
        elif functional == norm2_penal:
            jacobian = jac_norm2_penal
        elif functional == sobolev_norm_penal:
            jacobian = jac_sobolev_norm_penal
        elif functional == entropy_penal:
            jacobian =  jac_entropy_penal
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
        tolerance = 1e-6
    else:
        tolerance = tolerance
    if not max_iter:
        max_iter = 1000
    else: 
        max_iter = max_iter
    if not bounds:
        bound = scop.Bounds(0,np.inf)
    else:
        bound = bounds
    result = scop.minimize(functional,initial_guess,args=(alpha,n,trans_matrix_long,step_size),method='trust-constr',jac=jacobian,hess=hessian_method,constraints=[constraints],tol=tolerance,options={'verbose': 1,'maxiter': max_iter},bounds=bound)
    return result

#2-D functions

def vertices(n,k,step_size): #n is the box index
    import itertools
    vertex_dir = list(itertools.product([0,1],repeat=3))
    initialization = np.array(vertex_dir)*step_size - np.floor(np.ceil(k**(1/3))/2)*step_size
    x2 = np.floor(n/k**(2/3))
    x1 = np.floor((n - x2*k**(2/3))/k**(1/3))
    x0 = np.floor(n - x1*k**(1/3) - x2*k**(2/3))
    adjustment = np.array([x0,x1,x2])
    vertices = initialization + step_size*adjustment
    return vertices

def edge_check(p,start,end):
    check=[]
    for i in p:
        check.append((abs(i)-abs(start))==0 or (abs(i)-abs(start))==abs(abs(end)-abs(start)))
    return check

def outer_edge_check(p,start,end):
    return (abs(p)-abs(start))==0 or (abs(p)-abs(start))==abs(abs(end)-abs(start))

def is_point_vertex(point,step_size):
    check = point%step_size
    bools = check == 0
    val = sum(bools)
    return val==3
def is_point_in_box(point,vertices,interval):
    #possible time save is to limit the search range
    vert_point = vertices - point
    bools = vert_point == 0
    bool_sum = np.array([sum(bools[n])for n in range(len(bools))])
    indices = list(np.where(bool_sum > 1)[0])
    sums = []
    for i in indices:
        sums.append(sum(vert_point[i]))
    val = sum(np.array(sums)>0) < 2 and sum(np.array(sums)>0) >0
    return val
def get_all_box_vertex(point,vertex_collection,interval):
    #possible time save is to limit the search range
    vert_point = vertex_collection - point
    bools =  vert_point == 0
    bool_sum = np.array([np.max(np.sum(bools[n],axis=1)) for n in range(len(bools))])
    indices = np.where(bool_sum ==3)
    return list(indices[0])
def get_all_box_edge(point,vertex_collection,interval,step_size):
    vert_point = vertex_collection - point
    bool_sum = np.array([np.sum(np.sum(abs(vert_point[n]),axis=1)[np.sum(abs(vert_point[n]),axis=1)<step_size]) for n in range(len(vert_point))])
    indices = np.where(bool_sum == step_size)
    return indices
def get_all_box(point,vertex_collection,interval,step_size):
    if is_point_vertex(point,step_size):
        return get_all_box_vertex(point,vertex_collection,interval)
    else:
        return get_all_box_edge(point,vertex_collection,interval,step_size)
def search_range(index,n,k):
    val = k >=  max(0, index - 2*k**(2/3)) and k <= index +2*k**(2/3)
    return val
def get_index(point,interval,k):
    x=[]
    for p in point:
        x.append(max(np.where(interval>=p)[0][0]-1,0))
    index = x[0] + x[1]*k**(1/3) + x[2]*k**(2/3)
    return np.ceil(index)
def get_index(point,interval,k):
    x=[]
    if sum(point)<abs(interval[0])*3:
        for p in point:
            x.append(max(np.where(interval>=p)[0][0]-1,0))
    else:
        for p in point:
            x.append(max(np.where(interval>=p)[0][0],0))
    index = x[0] + x[1]*k**(1/3) + x[2]*k**(2/3)
    return np.ceil(index)
def get_index(point,interval,k):
    x=[]
    for p in point:
        x.append(max(np.where(interval>=p)[0][0]-1,0))
    index = x[0] + x[1]*k**(1/3) + x[2]*k**(2/3)
    return np.ceil(index)
def point_gen(n):
    x0 = np.random.uniform(-1,1,n)
    x1 = np.random.choice(np.arange(-1,1,0.25),n)
    x2 = np.random.choice(np.arange(-1,1,0.25),n)
def get_box_index(point,start,end,step_size,interval,ks):
    if (sum((abs(point)-abs(start))==0) == 3 or sum((abs(point)-abs(end))==0) == 3): #case for outer-vertex
        indices = np.array([get_index(point,interval,ks[2])],dtype=int) 
        return list(indices)
    elif ((sum((abs(point)-abs(start))==0) == 2 or sum((abs(point)-abs(end))==0) == 2) and sum((point%step_size)==0)==3): 
        #case for corner-outer-edge vertex
        if abs(point[1])-abs(start)==0 and abs(point[2])-abs(start)==0:
            indices = np.array([get_index(point,interval,ks[2])] * 2,dtype=int)
            adjustments = np.array([0,1],dtype=int)
            return list(indices+adjustments)
        elif  abs(point[0])-abs(start)==0 and abs(point[2])-abs(start)==0:
            indices = np.array([get_index(point,interval,ks[2])] * 2,dtype=int)
            adjustments = np.array([0,ks[0]],dtype=int)
            return list(indices+adjustments) 
        elif  abs(point[0])-abs(start)==0 and abs(point[1])-abs(start)==0:
            indices = np.array([get_index(point,interval,ks[2])] * 2,dtype=int)
            adjustments = np.array([0,ks[1]],dtype=int)
            return list(indices+adjustments)         
    elif ((sum((abs(point)-abs(start))==0) == 1 or sum((abs(point)-abs(end))==0) == 1) and sum((point%step_size)==0)==3): 
        #case for side-outer-edge vertex
        #needs cases for different axes
        if abs(point[1])-abs(start)==0:
            indices = np.array([get_index(point,interval,ks[2])] * 4,dtype=int)
            adjustments = np.array([0,1,ks[1],ks[1]+1],dtype=int)
            return list(indices+adjustments)
        elif abs(point[0])-abs(start)==0:
            indices = np.array([get_index(point,interval,ks[2])] * 4,dtype=int)
            adjustments = np.array([0,ks[0],ks[1],ks[1]+ks[0]],dtype=int)
            return list(indices+adjustments)
        elif abs(point[2])-abs(start)==0:
            indices = np.array([get_index(point,interval,ks[2])] * 4,dtype=int)
            adjustments = np.array([0,1,ks[0],ks[0]+1],dtype=int)
            return list(indices+adjustments)
    elif sum(point%step_size) == 0: #case for inner-vertex
        indices = np.array([get_index(point,interval,ks[2])] * 8,dtype=int)
        adjustments = np.array([0,1,ks[0],ks[0]+1,ks[1],ks[1]+1,+ks[0]+ks[1],ks[0]+ks[1]+1],dtype=int)
        return list(indices+adjustments)
    elif sum(edge_check(point,start,end))==2: #case for outermost-edge
        indices = np.array([get_index(point,interval,ks[2])],dtype=int) 
        return list(indices)
    elif (sum((abs(point)-abs(start))==0)==1 or sum((abs(point)-abs(end))==0)==1): #case for outer-edge
        if outer_edge_check(point[1],start,end): #change logic
            if point[0]%step_size==0 :
                indices = np.array([get_index(point,interval,ks[2])]*2,dtype=int)
                adjustments = np.array([0,1])
                return list(indices+adjustments)
            else:
                indices = np.array([get_index(point,interval,ks[2])]*2,dtype=int)
                adjustments = np.array([0,ks[1]])
                return list(indices+adjustments)                
        elif outer_edge_check(point[0],start,end):
            if point[1]%step_size==0:
                indices = np.array([get_index(point,interval,ks[2])]*2,dtype=int)
                adjustments = np.array([0,ks[0]])
                return list(indices+adjustments)
            else:
                indices = np.array([get_index(point,interval,ks[2])]*2,dtype=int)
                adjustments = np.array([0,ks[1]])
                return list(indices+adjustments)  
        elif outer_edge_check(point[2],start,end):
            if point[0]%step_size==0:
                indices = np.array([get_index(point,interval,ks[2])]*2,dtype=int)
                adjustments = np.array([0,1])
                return list(indices+adjustments)
            else:
                indices = np.array([get_index(point,interval,ks[2])]*2,dtype=int)
                adjustments = np.array([0,ks[0]])
                return list(indices+adjustments)  
    elif sum((point%step_size)==0)==2 : #case for inner-edge #needs more cases
        if point[0]%step_size !=0:
            indices = np.array([get_index(point,interval,ks[2])]*4,dtype=int)
            adjustments = np.array([0,ks[0],ks[1],ks[0]+ks[1]])
            return list(indices+adjustments)
        elif point[1]%step_size !=0:
            indices = np.array([get_index(point,interval,ks[2])]*4,dtype=int)
            adjustments = np.array([0,1,ks[1],ks[1]+1])
            return list(indices+adjustments)
        elif point[2]%step_size !=0:
            indices = np.array([get_index(point,interval,ks[2])]*4,dtype=int)
            adjustments = np.array([0,1,ks[0],ks[0]+1])
            return list(indices+adjustments)

def angle(v1,v2):
    dot = np.dot(v1[0:2],v2[0:2])
    det = v1[0]*v2[1] - v1[1]*v2[0]
    temp_cos = dot/np.linalg.norm(v1[0:2])/np.linalg.norm(v2[0:2])
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
    area = 0.5 * abs(points[0][0] * points[-1][1] - points[0][1] * points[-1][0])
    for i in range(len(points) - 1):
        area += 0.5 * abs(points[i][0] * points[i + 1][1] - points[i][1] * points[i + 1][0])
    return area

def ransample(n,pi,mu,sigma):
    x=np.zeros((n))
    y=np.zeros((n))
    z=np.zeros((n))
    k=np.random.choice(len(pi),n,p=pi,replace=True)
    for i in range(0,len(k)):
        x[i],y[i],z[i]=np.random.multivariate_normal(mu[k[i]],cov[k[i]],1).T
    return x,y,z

def big_L2d(sample_size,k):
    coord_mtx=np.zeros((sample_size,k))
    return coord_mtx

def create_L2d(sample,f_dimension,start,end,step,interval,ks):
    L=big_L2d(len(sample),f_dimension)
    b0s=interval
    b1s=interval
    b2s=interval
    for n in range(0,len(sample)):
        b0_based_intersections = []
        b1_based_intersections = []
        b2_based_intersections = []
        for i in b1s:
            for j in b2s:
                b0_int = (xy_sample[n,3] - i*xy_sample[n,1] - j*xy_sample[n,2])/xy_sample[n,0]
                if abs(b0_int) <= abs(start):
                    b0_based_intersections.append(tuple([b0_int,i,j]))
        for i in b0s:
            for j in b2s:
                b1_int = (xy_sample[n,3] - i*xy_sample[n,0] - j*xy_sample[n,2])/xy_sample[n,1]
                if abs(b1_int) <=  abs(start):
                    b1_based_intersections.append(tuple([i,b1_int,j]))
        for i in b0s:
            for j in b1s:
                b2_int = (xy_sample[n,3] - i*xy_sample[n,0] - j*xy_sample[n,1])/xy_sample[n,2]
                if abs(b2_int) <= abs(start):
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
    return L

def second_deriv_3d(f,step):
    f=f+10e-3
    f=f.reshape(int(np.ceil(len(f)**(1/3))),int(np.ceil(len(f)**(1/3))),int(np.ceil(len(f)**(1/3))))
    fgrad0=np.ravel(np.gradient(np.gradient(f,step)[0],step)[0])
    fgrad1=np.ravel(np.gradient(np.gradient(f,step)[1],step)[1])
    fgrad2=np.ravel(np.gradient(np.gradient(f,step)[2],step)[2])
    return fgrad0+fgrad1+fgrad2

def sobolev_norm_penal2d(f,alpha,n,L_mat_long,step):
    import numpy as np
    L_mat=L_mat_long.reshape(n,len(f))
    f[f<0]=0
    f=f+10e-3
    val=np.log(np.dot(L_mat,f))
    return -sum(val)/n + alpha*step**3*sum(f**2)+alpha*step**3*norm_fprime_3d(f,step)

def norm_fprime_3d(f,step):
    f=f.reshape(int(np.ceil(len(f)**(1/3))),int(np.ceil(len(f)**(1/3))),int(np.ceil(len(f)**(1/3))))
    fgrad=np.gradient(f,step)
    return sum(np.ravel((fgrad[0]**2+fgrad[1]**2+fgrad[2]**2)))

def jac_sobolev_penal2d(f,alpha,n,L_mat_long,step):
    import numpy as np
    f=f+10e-3
    L_mat=L_mat_long.reshape(n,len(f))
    denom=np.dot(L_mat,f)
    val=L_mat.T/denom
    return -val.T.sum(axis=0)/n+alpha*step**3*2*f-2*alpha*step**3*second_deriv_3d(f,step)

def rmle_2d(functional,alpha,trans_matrix,step_size,jacobian=None,initial_guess=None,hessian_method=None,constraints=None,tolerance=None,max_iter=None,bounds=None):
    n=len(trans_matrix)
    m=len(np.transpose(trans_matrix))
    trans_matrix_long = np.ravel(trans_matrix)
    if not initial_guess:
        initial_guess = np.zeros(m)+0.000001
    else:
        initial_guess = initial_guess
    if not jacobian:
        if functional == sobolev_norm_penal2d:
        	jacobian = jac_sobolev_penal2d
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
        tolerance = 1e-16
    else:
        tolerance = tolerance
    if not max_iter:
        max_iter = 1000
    else: 
        max_iter = max_iter
    if not bounds:
        bound = scop.Bounds(0,np.inf)
    else:
        bound = bounds
    result = scop.minimize(functional,initial_guess,args=(alpha,n,trans_matrix_long,step_size),method='trust-constr',jac=jacobian,hess=hessian_method,constraints=[constraints],tol=tolerance,options={'verbose': 1,'maxiter': max_iter},bounds=bound)
    return result

def plot_rmle(f,step_size=None,dim=None,grid_lims=None):
	if not dim:
		f_dim = len(np.shape(f))
	else:
		f_dim = dim
	if f_dim == 2:
		if not grid_lims and not step_size:
			plt.contour(f)
		else:
			if not step_size:
				X=np.arange(grid_lims[0],grid_lims[1],0.25)
				Y=np.arange(grid_lims[0],grid_lims[1],0.25)
				plt.contour(X,Y,f,colors='black')
			else:
				X=np.arange(grid_lims[0],grid_lims[1],step_size)
				Y=np.arange(grid_lims[0],grid_lims[1],step_size)
				plt.contour(X,Y,f,colors='black')
	else:
		if not grid_lims and not step_size:
			f01=np.sum(f,axis=1)
			f02=np.sum(f,axis=2)
			f12=np.sum(f,axis=0)
			plt.contour(f01,colors='black')
			plt.show()
			plt.contour(f02,colors='black')
			plt.show()
			plt.contour(f12,colors='black')
			plt.show()
		else:
			if not step_size:
				f01=np.sum(f,axis=1)*0.5
				f02=np.sum(f,axis=2)*0.5
				f12=np.sum(f,axis=0)*0.5
				X=np.arange(grid_lims[0],grid_lims[1],0.5)
				Y=np.arange(grid_lims[0],grid_lims[1],0.5)
				plt.contour(X,Y,f01,colors='black')
				plt.show()
				plt.contour(X,Y,f02,colors='black')
				plt.show()
				plt.contour(X,Y,f12,colors='black')
				plt.show()
			else:
				f01=np.sum(f,axis=1)*step_size
				f02=np.sum(f,axis=2)*step_size
				f12=np.sum(f,axis=0)*step_size
				X=np.arange(grid_lims[0],grid_lims[1],step_size)
				Y=np.arange(grid_lims[0],grid_lims[1],step_size)
				plt.contour(X,Y,f01,colors='black')
				plt.show()
				plt.contour(X,Y,f02,colors='black')
				plt.show()
				plt.contour(X,Y,f12,colors='black')
				plt.show()
			
