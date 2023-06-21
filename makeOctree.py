import os, sys, time
import numpy as np
import pickle
import json

import open3d as o3d
import trimesh

import dataclasses
from dataclasses import dataclass
import pyrallis
from pyrallis import field

import typing
from typing import Any, List, Dict

sys.path.append('octree_base/')
import octree_base.octree as octree
from visualise_octree import vis_octree
from utils import DotDict, make_print_also_log, PointsScaler, CenteredGrid, pointsTensor2octreeSigns, implicitFunc2mesh

@dataclass
class OctreeConfig:
    # Path to the input Point Cloud to reconstruct from. Can be either a single path, or a path that matches to all PCs to 
    # reconstruct from using UNIX stype pathname expansion using *. E.g. 
    # input_pc_path : str = '/media/chamin8TB/MultiresUnorientedNF/data/deep_geometric_prior_data/scans/gargoyle.ply' # or
    # input_pc_path : str = '/media/chamin8TB/MultiresUnorientedNF/data/deep_geometric_prior_data/scans/*.ply'
    input_pc_path : str = '/media/chamin8TB/MultiresUnorientedNF/data/deep_geometric_prior_data/scans/gargoyle.ply'
    shape_name : str = 'gargoyle' # Name of the shape
    dataset_name : str = 'SRB' # Name of the dataset
    # Where to put logs, output and saves
    log_dir: str = '/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/octree_labels2'

    recon_radius : float = 1.1 # After normalising the PC to the unit sphere, what radius cube to do the reconstruction on
    initial_depth : int = 3 # depth to build the octree to
    final_depth : int = 7 # depth to build the octree to
    is2D : bool = False # whether to build and octree in 2D or 3D, should be same as 

    show_vis : bool = False # whether to show visualisations
    save_pkl : bool = False # whether to serialise octree to pkl file

    use_wandb : bool = False # Whether to log using wandb

    notes : str = "nothing much" # Notes to log
    extra_tags : List[str] = field(default_factory=lambda : []) # Extra tags to add
    extra : dict = field(default_factory=lambda : DotDict()) # A place to add extra values

    def __post_init__(self):
        self.shape_log_dir = os.path.join(self.log_dir, self.shape_name)

def make_mesh(ot, current_depth, cfg, grid_obj, scale_obj):
    # cfg, grid_obj, scale_obj
    implicit_func = lambda points : pointsTensor2octreeSigns(points, ot, cfg.recon_radius, current_depth)
    mesh = implicitFunc2mesh(grid_obj, implicit_func, scale_obj.unscale_points, chunk_size = 1000000, use_tqdm=False)
    return mesh

def recon_shape(cfg):
    # # parse the cfg
    # cfg = pyrallis.parse(config_class=OctreeConfig)

    # cfg.show_vis = True
    # cfg.show_vis = False
    # cfg.save_pkl = False
    # cfg.save_pkl = True
    # cfg.is2D = False
    # cfg.extra.d = 1

    assert cfg.recon_radius >= 1.0, f"recon_radius should be >= 1, but have value {cfg.recon_radius:.5f}"

    # import pdb; pdb.set_trace()
    tags = [cfg.dataset_name, cfg.shape_name, "2D" if cfg.is2D else "3D", f"FinalDepth={cfg.final_depth}", *cfg.extra_tags]

    # after all changes to the cfg
    if cfg.use_wandb:
        import wandb
        wandb.init(
            project = f"OG-INR-octrees_{cfg.dataset_name}_dataset",
            notes = cfg.notes,
            tags = tags,
            config = dataclasses.asdict(cfg)
        )

    # make the log directory
    os.makedirs(cfg.shape_log_dir, exist_ok=True)

    # set print 
    log_filepath = os.path.join(cfg.shape_log_dir, 'out.log')
    print = make_print_also_log(log_filepath) # make print also save to log file
    print(cfg)
    # print(dataclasses.asdict(cfg))
    print("Tags:", tags)

    def log_dict(cost, depth, ot, other={}):
        return {"cost": cost, "depth": depth, "numSurfaceLeaves": ot.numSurfaceLeaves, "numNonSurfaceLeaves": ot.numNonSurfaceLeaves, 
                                "numInsideLeaves": ot.labels.sum(), "numOutsideLeaves": (ot.labels==0).sum(), 
                                "Inside%": ot.labels.mean(), "Outside%": (ot.labels==0).mean(), **other}

    ################################
    ## Getting Data
    ################################
    vis_time = 0 # remove time used for visualisation and other non-algorithmic processes
    initial_time = time.time()
    t0 = time.time()
    print('Starting...')

    # read point cloud
    o3d_point_cloud = o3d.io.read_point_cloud(cfg.input_pc_path)
    mnfld_pnts = np.asarray(o3d_point_cloud.points, dtype=np.float32)

    scale_obj = PointsScaler(mnfld_pnts)
    mnfld_pnts = scale_obj.scaled_points
    
    scaling_savepath = os.path.join(cfg.shape_log_dir, 'scaling.npz')
    np.savez(scaling_savepath, cp=scale_obj.cp, max_norm=scale_obj.max_norm)
    print(f"Point cloud scaled, scaling parameters saved to {scaling_savepath}")

    # Make a grid obj for saving meshes
    grid_obj = CenteredGrid(2 if cfg.is2D else 3, grid_res=128, radius=cfg.recon_radius)

    t1 = time.time()

    vis_time += time.time() - t1
    print('Loaded PC', time.time()-t0 - vis_time); t0 = time.time()

    ################################
    ## Building Tree
    ################################
    np.random.seed(0)

    depth = cfg.initial_depth
    num_extensions = cfg.final_depth - cfg.initial_depth
    print(f"Depth = {cfg.initial_depth}, Shape = {cfg.shape_name}")

    ot = octree.py_createTree(depth, mnfld_pnts, 
                                2*cfg.recon_radius, # cube length
                                -cfg.recon_radius, # min_x
                                -cfg.recon_radius, # min_y
                                -cfg.recon_radius, # min_z
                                cfg.is2D)
    print('after octree creation ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
    # octree.py_computeAllDists(ot)
    # print('after computing all distances ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
    # cost = octree.py_computeCost(ot)
    # print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
    octree.py_grow_and_expand(ot, depth, mnfld_pnts)
    print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
    octree.py_computeAllDists(ot)
    print('after computing all distances ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
    cost = octree.py_computeCost(ot)
    print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
    if cfg.use_wandb: wandb.log(log_dict(cost, depth, ot, other={}))
    if cfg.show_vis:
        t1 = time.time()
        labels = ot.labels
        vis_octree(mnfld_pnts, ot, labels)
        vis_time += time.time() - t1; t0 = time.time()
    # octree.py_grow_if_cost_reduce(ot)
    # print('after growing if cost reduces ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
    # cost = octree.py_computeCost(ot)
    # print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
    # if cfg.show_vis:
    #     t1 = time.time()
    #     labels = ot.labels
    #     vis_octree(mnfld_pnts, ot, labels)
    #     vis_time += time.time() - t1; t0 = time.time()
    for _ in range(3):
        octree.py_makeSingleMove(ot)
        print('after single move ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        cost = octree.py_computeCost(ot)
        print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        octree.py_grow(ot)
        print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        cost = octree.py_computeCost(ot)
        print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        if cfg.use_wandb: wandb.log(log_dict(cost, depth, ot, other={}))
    for _ in range(5):
        num_changed = octree.py_makeTwoStepMove(ot)
        print('after two step move ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        cost = octree.py_computeCost(ot)
        print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        octree.py_grow(ot)
        print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        cost = octree.py_computeCost(ot)
        print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        if cfg.use_wandb: wandb.log(log_dict(cost, depth, ot, other={}))
        if num_changed == 0:
            break
    for _ in range(3):
        octree.py_makeSingleMove(ot)
        print('after single move ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        cost = octree.py_computeCost(ot)
        print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        octree.py_grow(ot)
        print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        cost = octree.py_computeCost(ot)
        print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        if cfg.use_wandb: wandb.log(log_dict(cost, depth, ot, other={}))
    octree.py_grow(ot)
    print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
    cost = octree.py_computeCost(ot)
    print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
    if cfg.use_wandb: wandb.log(log_dict(cost, depth, ot, other={}))
    if cfg.show_vis:
        t1 = time.time()
        labels = ot.labels
        vis_octree(mnfld_pnts, ot, labels)
        vis_time += time.time() - t1; t0 = time.time()
    if cfg.save_pkl:
        t1 = time.time()
        savepath = os.path.join(cfg.shape_log_dir, '{}_depth_{}.pkl'.format(cfg.shape_name, depth))
        data = octree.py_serialise_tree(ot)
        print(f"Saving to {savepath}")
        with open(savepath, 'wb') as fp:
            # pickle.dump((depth, mnfld_pnts, cp, max_norm, cfg.initial_depth, ot.labels), fp)
            pickle.dump(data, fp)
        vis_time += time.time() - t1; t0 = time.time()    
    
    mesh = make_mesh(ot, depth, cfg, grid_obj, scale_obj)
    savepath = os.path.join(cfg.shape_log_dir, '{}_depth_{}.ply'.format(cfg.shape_name, depth))
    mesh.export(savepath)


    for j in range(num_extensions):
        depth += 1
        print('Depth is ', depth)
        octree.py_extend_tree(ot, depth, mnfld_pnts)
        print('after octree extending ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        # octree.py_computeAllDists(ot)
        # print('after computing all distances ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        # cost = octree.py_computeCost(ot)
        # print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        # if cfg.show_vis:
        #     t1 = time.time()
        #     labels = ot.labels
        #     vis_octree(mnfld_pnts, ot, labels)
        #     vis_time += time.time() - t1; t0 = time.time()
        octree.py_grow_and_expand(ot, depth, mnfld_pnts)
        print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        octree.py_computeAllDists(ot)
        print('after computing all distances ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        cost = octree.py_computeCost(ot)
        print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        # if cfg.show_vis:
        #     t1 = time.time()
        #     labels = ot.labels
        #     vis_octree(mnfld_pnts, ot, labels)
        #     vis_time += time.time() - t1; t0 = time.time()
        if cfg.use_wandb: wandb.log(log_dict(cost, depth, ot, other={}))

        for _ in range(3):
            octree.py_makeSingleMove(ot)
            print('after single move ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
            cost = octree.py_computeCost(ot)
            print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
            octree.py_grow(ot)
            print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
            cost = octree.py_computeCost(ot)
            print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
            if cfg.use_wandb: wandb.log(log_dict(cost, depth, ot, other={}))
        # if cfg.show_vis:
        #     t1 = time.time()
        #     labels = ot.labels
        #     vis_octree(mnfld_pnts, ot, labels)
        #     vis_time += time.time() - t1; t0 = time.time()
        print("Making 2 step move of length 2 ######################################################")
        for _ in range(5):
            num_changed = octree.py_makeTwoStepMove(ot,2)
            print('after two step move ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
            cost = octree.py_computeCost(ot)
            print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
            # octree.py_grow(ot)
            # print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
            # cost = octree.py_computeCost(ot)
            # print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
            if cfg.use_wandb: wandb.log(log_dict(cost, depth, ot, other={}))
            if num_changed == 0:
                break
        print("Making 2 step move of length 11 ######################################################")
        for _ in range(5):
            num_changed = octree.py_makeTwoStepMove(ot,10)
            print('after two step move ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
            cost = octree.py_computeCost(ot)
            print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
            # octree.py_grow(ot)
            # print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
            # cost = octree.py_computeCost(ot)
            # print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
            if cfg.use_wandb: wandb.log(log_dict(cost, depth, ot, other={}))
            if num_changed == 0:
                break
        # print("Making 2 step move of length 30 ######################################################")
        # for _ in range(5):
        #     num_changed = octree.py_makeTwoStepMove(ot,30)
        #     print('after two step move ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        #     cost = octree.py_computeCost(ot)
        #     print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        #     # octree.py_grow(ot)
        #     # print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        #     # cost = octree.py_computeCost(ot)
        #     # print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        #     if num_changed == 0:
        #         break
        print("Making 2 step move of length 10000 ######################################################")
        num_changed = octree.py_makeTwoStepMove(ot,10000)
        print('after two step move ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        cost = octree.py_computeCost(ot)
        print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        if cfg.use_wandb: wandb.log(log_dict(cost, depth, ot, other={}))
        print("Making 2 step move of length 10 ######################################################")
        num_changed = octree.py_makeTwoStepMove(ot,10)
        print('after two step move ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        cost = octree.py_computeCost(ot)
        print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        if cfg.use_wandb: wandb.log(log_dict(cost, depth, ot, other={}))

        
        # for _ in range(5):
        #     num_changed = octree.py_makeTwoStepMove(ot)
        #     print('after two step move ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        #     cost = octree.py_computeCost(ot)
        #     print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        #     # octree.py_grow(ot)
        #     # print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        #     # cost = octree.py_computeCost(ot)
        #     # print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        #     if num_changed == 0:
        #         break
        # for _ in range(5):
        #     num_changed = octree.py_makeTwoStepMove(ot, 1)
        #     print('after two step move ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        #     cost = octree.py_computeCost(ot)
        #     print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        #     octree.py_grow(ot)
        #     print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
        #     cost = octree.py_computeCost(ot)
        #     print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
        #     if num_changed == 0:
        #         break
        # if cfg.show_vis:
        #     t1 = time.time()
        #     labels = ot.labels
        #     vis_octree(mnfld_pnts, ot, labels)
        #     vis_time += time.time() - t1; t0 = time.time()
        for _ in range(2):
            octree.py_makeSingleMove(ot)
            print('after single move ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
            cost = octree.py_computeCost(ot)
            print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
            # octree.py_grow(ot)
            # print('after outside growing ({:.5f}s)'.format(time.time()-t0)); t0 = time.time()
            # cost = octree.py_computeCost(ot)
            # print('\t Computed cost={:.5f}, ({:.5f}s)'.format(cost, time.time()-t0)); t0 = time.time()
            if cfg.use_wandb: wandb.log(log_dict(cost, depth, ot, other={}))
        print('Octree Creation and Labelling up till depth {}: ({:.5f}s)'.format(depth, time.time()-initial_time - vis_time))
        # if cfg.show_vis:
        #     t1 = time.time()
        #     labels = ot.labels
        #     vis_octree(mnfld_pnts, ot, labels)
        #     vis_time += time.time() - t1; t0 = time.time()
        if cfg.show_vis:
            t1 = time.time()
            labels = ot.labels
            vis_octree(mnfld_pnts, ot, labels)
            vis_time += time.time() - t1; t0 = time.time()
        if cfg.save_pkl:
            t1 = time.time()
            savepath = os.path.join(cfg.shape_log_dir, '{}_depth_{}.pkl'.format(cfg.shape_name, depth))
            data = octree.py_serialise_tree(ot)
            print(f"Saving to {savepath}")
            with open(savepath, 'wb') as fp:
                # pickle.dump((depth, mnfld_pnts, cp, max_norm, cfg.initial_depth, ot.labels), fp)
                pickle.dump(data, fp)
            vis_time += time.time() - t1; t0 = time.time()
        mesh = make_mesh(ot, depth, cfg, grid_obj, scale_obj)
        savepath = os.path.join(cfg.shape_log_dir, '{}_depth_{}.ply'.format(cfg.shape_name, depth))
        mesh.export(savepath)

    print('Octree Creation and Labelling: ({:.5f}s)'.format(time.time()-initial_time - vis_time)); t0 = time.time()
    if cfg.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    # SRB
    # scans: /media/chamin8TB/MultiresUnorientedNF/data/deep_geometric_prior_data/scans/*.ply
    # gt: /media/chamin8TB/MultiresUnorientedNF/data/deep_geometric_prior_data/ground_truth/*.xyz

    # ShapeNet
    # scans: /media/chamin8TB/MultiresUnorientedNF/data/NSP_dataset/*/*.ply
    # gt: /media/chamin8TB/MultiresUnorientedNF/data/ShapeNetNSP/*/*/pointcloud.npz
    # gt: /media/chamin8TB/MultiresUnorientedNF/data/ShapeNetNSP/*/*/points.npz

    # SRB to depth 6
    # python makeOctree.py --input_pc_path=/media/chamin8TB/MultiresUnorientedNF/data/deep_geometric_prior_data/scans/*.ply --dataset_name=SRB --log_dir=/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/octree_labels3 --final_depth=6 --show_vis=False --save_pkl=True --use_wandb=False
    # depth 7
    # python makeOctree.py --input_pc_path=/media/chamin8TB/MultiresUnorientedNF/data/deep_geometric_prior_data/scans/*.ply --dataset_name=SRB --log_dir=/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/SRB --final_depth=7 --show_vis=False --save_pkl=True --use_wandb=False

    # NewSRB Real object
    # scans: /media/chamin8TB/NewSRBDataset/real_object_scan/*.ply
    # gt: /media/chamin8TB/NewSRBDataset/real_object_GT/real_gt/*.xyz
    # scans: /media/chamin8TB/NewSRBDataset/PerfectPLYs/Uniform/complex/*.ply
    # scans: /media/chamin8TB/NewSRBDataset/PerfectPLYs/Uniform/ordinary/*.ply
    # scans: /media/chamin8TB/NewSRBDataset/PerfectPLYs/Uniform/simple/*.ply
    # scans: /media/chamin8TB/NewSRBDataset/SCUT-SRB/Perfect/*.ply
    # scans: /media/chamin8TB/NewSRBDataset/SCUT-SRB/real_object_scan

    # to depth 7
    # python makeOctree.py --input_pc_path=/media/chamin8TB/NewSRBDataset/SCUT-SRB/Perfect/*.ply --dataset_name=NewSRB-perfet --log_dir=/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/octree_perfet --final_depth=7 --show_vis=False --save_pkl=True --use_wandb=False
    # to depth 7
    # python makeOctree.py --input_pc_path=/media/chamin8TB/MultiresUnorientedNF/data/NSP_dataset/*/*.ply --dataset_name=ShapeNet --log_dir=/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/shapenet --final_depth=7 --show_vis=False --save_pkl=True --use_wandb=False


    # python makeOctree.py --input_pc_path=/media/chamin8TB/NewSRBDataset/SCUT-SRB/Perfect/*.ply --dataset_name=NewSRB-perfet --log_dir=/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/octree_perfectv2 --initial_depth=5 --final_depth=7 --show_vis=False --save_pkl=True --use_wandb=False
    # python makeOctree.py --input_pc_path=/media/chamin8TB/NewSRBDataset/SCUT-SRB/real_object_scan/*.ply --dataset_name=NewSRB-real_object --log_dir=/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/octree_real_objectv2 --initial_depth=5 --final_depth=7 --show_vis=False --save_pkl=True --use_wandb=False

    # python makeOctree.py --input_pc_path=/media/chamin8TB/MultiresUnorientedNF/data/NSP_dataset/*/*.ply --dataset_name=ShapeNetd6 --log_dir=/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/shapenetd6 --final_depth=6 --show_vis=False --save_pkl=True --use_wandb=False
    # parse the cfg
    cfg = pyrallis.parse(config_class=OctreeConfig)

    # input pc path must be a path to a single PC, or be a 
    if os.path.exists(cfg.input_pc_path):
        # single path given
        print('Input PC path exists')
        recon_shape(cfg)
    else:
        # multiple paths specified
        print('Input PC path does not exist, interpretting as multiple paths')
        initial_input_pc_path = cfg.input_pc_path
        initial_log_dir = cfg.log_dir
        initial_shape_name = cfg.shape_name
        import re, glob
        paths = glob.glob(initial_input_pc_path)
        for shape_path in paths:
            # find all the groups, i.e. if initial path is /media/chamin8TB/MultiresUnorientedNF/data/NSP_dataset/*/*.ply and
            # a matched path is /media/chamin8TB/MultiresUnorientedNF/data/NSP_dataset/cabinet/b278c93f18480b362ea98d69e91ba870.ply
            # then groups is where the * matches, i.e. ('cabinet', 'b278c93f18480b362ea98d69e91ba870')
            pattern = initial_input_pc_path.replace('*','(.*)')
            groups = re.compile(pattern).match(shape_path).groups()
            assert len(groups) >= 1, groups
            shape_name = groups[-1]
            cfg.input_pc_path = shape_path
            cfg.shape_name = shape_name
            cfg.log_dir = os.path.join(initial_log_dir, *groups[:-1])
            cfg.shape_log_dir = os.path.join(cfg.log_dir, cfg.shape_name)
            print(cfg.input_pc_path)
            print(cfg.shape_log_dir)
            print()
            recon_shape(cfg)

