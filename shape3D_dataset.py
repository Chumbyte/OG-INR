import os, sys, time
import numpy as np
import pickle

import torch
import torch.utils.data as data

import open3d as o3d
import trimesh
import scipy.spatial as spatial
import point_cloud_utils as pcu

sys.path.append('octree_base/')
import octree_base.octree as octree
from visualise_octree import vis_octree
from utils import DotDict, make_print_also_log, PointsScaler, CenteredGrid, pointsTensor2octreeSigns, implicitFunc2mesh

class ReconDataset(data.Dataset):
    def __init__(self, shape_name, pcd_path, octree_pkl_path, octree_depth, scaling_path, batch_size, grid_res=128, grid_radius=1.1, dataset_length=None):
        self.shape_name = shape_name
        self.pcd_path = pcd_path # path to the point cloud
        self.octree_pkl_path = octree_pkl_path
        self.octree_depth = octree_depth
        self.scaling_path = scaling_path
        self.batch_size = batch_size
        self.grid_res = grid_res # g_res
        self.grid_radius = grid_radius
        self.is2D = False
        self.dataset_length = 1000 if dataset_length is None else dataset_length
        
        print('Loading PCD from {}'.format(self.pcd_path)); t0 = time.time()
        # load data
        self.o3d_point_cloud = o3d.io.read_point_cloud(self.pcd_path)        
        # Returns points on the manifold
        self.pcd_points = np.asarray(self.o3d_point_cloud.points, dtype=np.float32)
        self.pcd_normals = np.asarray(self.o3d_point_cloud.normals, dtype=np.float32) # might be empty
        # center and scale point cloud
        scaling = np.load(scaling_path)
        self.cp = scaling['cp']
        self.max_norm = scaling['max_norm']
        self.scale_obj = PointsScaler(self.pcd_points, self.cp, self.max_norm)
        self.pcd_points = self.scale_obj.scaled_points
        print('PCD loaded and scaled ({:.5f}s)'.format(time.time()-t0))

        print('Making grid'); t0 = time.time()
        # Make a grid obj for saving meshes
        self.grid_obj = CenteredGrid(2 if self.is2D else 3, grid_res=self.grid_res, radius=self.grid_radius)
        print('Grid made ({:.5f}s)'.format(time.time()-t0))

        print(f"Loading octree from {octree_pkl_path}"); t0 = time.time()
        with open(octree_pkl_path, 'rb') as fp:
            data = pickle.load(fp)
        ot = octree.py_deserialise_tree(data, self.grid_radius*2, -self.grid_radius, -self.grid_radius, -self.grid_radius, self.is2D)
        print('Octree loaded ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()

        self.num_nonsurf = ot.numNonSurfaceLeaves
        self.num_inside = ot.labels.sum()
        self.num_outside = self.num_nonsurf - self.num_inside
        self.num_surf = ot.numSurfaceLeaves
        print(f"# inside leafs {self.num_inside}, # outside leafs {self.num_outside}, # surface leafs {self.num_surf}")
        
        self.surf_lvs_centers = []
        for node in ot.surfaceLeaves:
            l = node.length
            x,y,z = node.top_pos_x, node.top_pos_y, node.top_pos_z
            self.surf_lvs_centers.append((x+l/2, y+l/2, z+l/2))
        self.surf_lvs_centers = np.array(self.surf_lvs_centers, dtype=np.float32)
        self.surf_leaf_length = l # should all be the same length

        self.nonsurf_lvs_centers = []
        self.nonsurf_lvs_lengths = []
        for node in ot.nonSurfaceLeaves:
            l = node.length
            x,y,z = node.top_pos_x, node.top_pos_y, node.top_pos_z
            self.nonsurf_lvs_centers.append((x+l/2, y+l/2, z+l/2))
            self.nonsurf_lvs_lengths.append(l)
        self.nonsurf_lvs_centers = np.array(self.nonsurf_lvs_centers, dtype=np.float32)
        self.nonsurf_lvs_lengths = np.array(self.nonsurf_lvs_lengths, dtype=np.float32)
        print('({:.5f}s)'.format(time.time()-t0)); t0 = time.time()

        surf_lv_signs = np.full(self.num_surf, -1, dtype=np.int32)
        self.nonsurf_lv_signs = -(ot.labels*2-1).astype(np.int32) # (Ni+No,)

        # self.leaf_centers = np.concatenate([self.surf_lvs_centers,self.nonsurf_lvs_centers], axis=0)
        # self.leaf_signs = np.concatenate([surf_lv_signs, nonsurf_lv_signs], axis=0)
        self.leaf_centers = self.nonsurf_lvs_centers
        self.leaf_signs = self.nonsurf_lv_signs
        print('Leaf centers and signs made ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()

        self.grid_points = self.grid_obj.grid_points_flattened.numpy() # (g_res*g_res, 2) or (g_res*g_res*g_res,3)
        # use pcu's KNN which uses nanoflann's very fast approx nn
        self.dists_gp_to_pcd, self.corrs_gp_to_pcd = pcu.k_nearest_neighbors(self.grid_points, self.pcd_points, k=1)
        print('Distances from grid points to input PCD made ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        self.gp_signs = pointsTensor2octreeSigns(torch.tensor(self.grid_points), ot, self.grid_radius, self.octree_depth)
        print('Signs for grid points determined from octree ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        self.gp_approx_sdf = self.dists_gp_to_pcd * self.gp_signs

        # use pcu's KNN which uses nanoflann's very fast approx nn
        self.dists_lc_to_pcd, self.corrs_lc_to_pcd = pcu.k_nearest_neighbors(self.leaf_centers, self.pcd_points, k=1)
        print('Distances from leaf centers to input PCD made ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        self.lc_signs = pointsTensor2octreeSigns(torch.tensor(self.leaf_centers), ot, self.grid_radius, self.octree_depth)
        print('Signs for leaf centers determined from octree ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        self.lc_approx_sdf = self.dists_lc_to_pcd * self.lc_signs

        print('data loaded'); t0 = time.time()
        # import pdb; pdb.set_trace()

    def __len__(self):
        # there is no clear length to the dataset
        # return 10000
        return self.dataset_length
    

    def __getitem__(self, index):
        # get input pcd points for the batch
        pcd_batch_idxes = np.random.permutation(self.pcd_points.shape[0])[:self.batch_size]
        batch_pcd_points = self.pcd_points[pcd_batch_idxes]

        # make domain samples as 0.25 leaf centers, 0.75 grid points
        num_lc = min(self.batch_size // 4, self.num_nonsurf)
        num_gp = self.batch_size - num_lc
        lc_batch_idxes = np.random.permutation(self.num_nonsurf)[:num_lc]
        batch_leaf_centers = self.nonsurf_lvs_centers[lc_batch_idxes]               # (num_lc, 3)
        batch_leaf_center_df = self.dists_lc_to_pcd[lc_batch_idxes]
        batch_leaf_center_sdf = self.lc_approx_sdf[lc_batch_idxes]
        gp_batch_idxes = np.random.permutation(self.grid_points.shape[0])[:num_gp]
        batch_grid_points = self.grid_points[gp_batch_idxes]
        batch_grid_point_df = self.dists_gp_to_pcd[gp_batch_idxes]
        batch_grid_point_sdf = self.gp_approx_sdf[gp_batch_idxes]
        domain_points = np.concatenate([batch_leaf_centers, batch_grid_points], axis=0) # (bs, 3)
        domain_df = np.concatenate([batch_leaf_center_df, batch_grid_point_df], axis=0) # (bs, )
        domain_sdf = np.concatenate([batch_leaf_center_sdf, batch_grid_point_sdf], axis=0) # (bs, )

        # make non-surface leaf samples: randomly sample 10 points within num_ls leaves, making bs//12*10 ~= 5/6*bs points
        num_ls = min(self.batch_size // 12, self.num_nonsurf)
        ls_batch_idxes = np.random.permutation(self.num_nonsurf)[:num_ls]
        batch_leaf_samples = self.nonsurf_lvs_centers[ls_batch_idxes,None,:] # (num_ls, 1, 3)
        batch_leaf_lengths = self.nonsurf_lvs_lengths[ls_batch_idxes,None,None] # (num_ls, 1, 1)
        batch_leaf_samples = batch_leaf_samples + (np.random.rand(num_ls,10,3).astype(np.float32)-0.5) * batch_leaf_lengths
        batch_leaf_samples = batch_leaf_samples.reshape(-1, 3)
        batch_leaf_sample_signs = np.tile(self.nonsurf_lv_signs[ls_batch_idxes][:,None], (1,10)).reshape(-1)

        # make surface leaf samples: randomly sample 10 points within num_sls leaves, making bs//60*10 ~= 1/6*bs points
        num_sls = min(self.batch_size // 60, self.num_surf)
        sls_batch_idxes = np.random.permutation(self.num_surf)[:num_sls]
        batch_surf_leaf_samples = self.surf_lvs_centers[sls_batch_idxes,None,:] # (num_sls, 1, 3)
        batch_surf_leaf_samples = batch_surf_leaf_samples + (np.random.rand(num_sls,10,3).astype(np.float32)-0.5) * self.surf_leaf_length
        batch_surf_leaf_samples = batch_surf_leaf_samples.reshape(-1, 3)

        return {'pcd_points': torch.Tensor(batch_pcd_points),
                'domain_points': torch.Tensor(domain_points),
                'domain_approx_df': torch.Tensor(domain_df),
                'domain_approx_sdf': torch.Tensor(domain_sdf),
                'nonsurf_leaf_samples': torch.Tensor(batch_leaf_samples),
                'nonsurf_leaf_sample_signs': torch.Tensor(batch_leaf_sample_signs),
                'surf_leaf_samples': torch.Tensor(batch_surf_leaf_samples),
                }

if __name__ == '__main__':
    # shape_name, pcd_path, octree_pkl_path, scaling_path, batch_size, grid_res=128, grid_radius=1.1
    shape_name = 'gargoyle'
    pcd_path = '/media/chamin8TB/MultiresUnorientedNF/data/deep_geometric_prior_data/scans/gargoyle.ply'
    octree_pkl_path = '/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/SRBv2/gargoyle/gargoyle_depth_7.pkl'
    octree_depth = 7
    scaling_path = '/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/SRBv2/gargoyle/scaling.npz'
    batch_size = 15000
    grid_res = 256
    grid_radius = 1.1
    ds = ReconDataset(shape_name, pcd_path, octree_pkl_path, octree_depth, scaling_path, batch_size, grid_res, grid_radius)
    res = ds[0]
    import pdb; pdb.set_trace()

