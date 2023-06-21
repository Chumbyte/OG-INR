import os, sys, time
import pickle
import typing
from tqdm import tqdm
import builtins as __builtin__

import numpy as np
import torch
from torch.autograd import grad

import trimesh
from scipy.spatial import cKDTree as KDTree
from skimage import measure

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __name__ = 'DotDict'
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

def make_print_also_log(log_filepath):
    def print(*args, **kwargs):
        __builtin__.print(*args, **kwargs)
        with open(log_filepath, 'a') as fp:
            __builtin__.print(*args, file=fp, **kwargs)
    return print

def count_parameters(model):
    #count the number of parameters in a given model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CenteredGrid():
    def __init__(self, grid_dims, grid_res=128, radius=1):
        # Make a grid in n dimensions. It spans a hypercube centred at the origin with lengths of 2*radius in each 
        # dimension, and has grid_res points in each dimension. Thus there are grid_res**grid_dims points in the grid.
        # Short names are n, g_res, g_rad
        self.grid_dims = grid_dims # n=2 or n=3
        self.grid_res = grid_res # g_res is usually a power of 2: 128, 256, 512 etc.
        self.radius = radius
        assert self.grid_dims == 2 or self.grid_dims == 3, self.grid_dims
        self.grid_range = np.stack([[-1.0*radius,1.0*radius] for _ in range(self.grid_dims)]) # (n,2)

        self.axes = [np.linspace(self.grid_range[i,0], self.grid_range[i,1], grid_res) for i in range(self.grid_dims)] # list of n (g_res,)
        grid_points_np = np.stack(np.meshgrid(*self.axes, indexing='ij'), axis=-1) # (g_res,g_res,2) or (g_res,g_res,g_res,3)
        self.grid_spacing = self.axes[0][1] - self.axes[0][0]

        self.grid_points = torch.tensor(grid_points_np, dtype=torch.float32) # (g_res,g_res,2) or (g_res,g_res,g_res,3), i.e. an (x,y[,z]) at each x,y[,z] point
        self.grid_points_flattened = self.grid_points.reshape(-1,self.grid_dims) # (g_res*g_res, 2) or (g_res*g_res*g_res,3)

class PointsScaler():
    def __init__(self, initial_points, cp=None, max_norm = None):
        # assert torch.is_tensor(initial_points), type(initial_points)
        assert len(initial_points.shape) == 2 and initial_points.shape[-1] in [2,3], initial_points.shape

        if cp is None:
            cp = initial_points.mean(axis=0, keepdims=True)
        scaled_points = initial_points - cp
        if max_norm is None:
            max_norm = np.linalg.norm(scaled_points, axis=-1).max(-1)
        scaled_points /= max_norm

        self.initial_points = initial_points
        self.cp = cp
        self.max_norm = max_norm
        self.scaled_points = scaled_points
    
    def scale_points(self, points):
        return (points - self.cp) / self.max_norm
    
    def unscale_points(self, points):
        return points*self.max_norm + self.cp

def pointsTensor2index3D(points, g_rad, g_res):
    # points should be a torch Tensor of shape (num_points, 2 or 3)
    assert torch.is_tensor(points), type(points)
    assert len(points.shape) == 2 and points.shape[-1] in [2,3], points.shape

    # normalise points from [-g_rad,g_rad]^n to [0,2*g_rad]^n, then to [0,1]^n, 
    # then to [0, g_res]^n, then to {0,1,...,g_res-1}^n
    return ((points+g_rad)*(g_res)/(2*g_rad)).floor().int().clamp(0,g_res-1)

def pointsTensor2octreeSigns(points, ot, ot_rad, ot_depth):
    # points should be a torch Tensor of shape (num_points, 2 or 3)
    assert torch.is_tensor(points), type(points)
    assert len(points.shape) == 2 and points.shape[-1] in [2,3], points.shape
    res = 2**ot_depth

    index3ds = pointsTensor2index3D(points, ot_rad, res).numpy()
    signs = ot.signsFromIndex3ds(index3ds, ot_depth) # returns an int list
    signs = np.array(signs, dtype = np.int32)
    return signs # (num_points, )

def implicitFunc2mesh(grid_obj, implicit_func, unscaling_func, chunk_size = 100000, use_tqdm=True):
    # grid_obj is an instance of CenteredGrid
    # implicit_func takes a pointsTensor (num_points, 2 or 3) and returns a value Tensor (num_points, )
    # unscaling func takes scaled points and unscales them
    points = grid_obj.grid_points_flattened

    z = []
    if use_tqdm:
        generator = tqdm(torch.split(points, chunk_size, dim=0))
    else:
        generator = torch.split(points, chunk_size, dim=0)
    for pnts in generator:
        # pnts: (chunk_size, 3)
        vals = implicit_func(pnts)
        if torch.is_tensor(vals):
            vals = vals.cpu().numpy()
        z.append(vals)
    z = np.concatenate(z, axis=0) # (num_pnts, )
    z = z.reshape(grid_obj.grid_res, grid_obj.grid_res, grid_obj.grid_res) # (g_res, g_res, g_res)

    verts, faces, normals, values = measure.marching_cubes(volume=z, level=0.0, 
                                spacing=(grid_obj.grid_spacing, grid_obj.grid_spacing, grid_obj.grid_spacing))

    verts = verts + np.array([-grid_obj.radius, -grid_obj.radius, -grid_obj.radius])
    verts = unscaling_func(verts)
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=values, validate=True)
    return mesh

