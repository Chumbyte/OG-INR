import os,sys,time
import numpy as np
import matplotlib.pyplot as plt
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import dataclasses
from dataclasses import dataclass
import pyrallis
from pyrallis import field

import typing
from typing import Any, List, Dict

# Put the project path in sys.path
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

import shape3D_dataset
from INR_model import NeuralFieldModel
from losses import OGINRLoss

from utils import DotDict, make_print_also_log, count_parameters , implicitFunc2mesh


# python train_INR.py --log_dir=/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/SRB_inr_temp --octree_path=/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/SRBv2/gargoyle/gargoyle_depth_7.pkl --scaling_path=/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/SRBv2/gargoyle/scaling.npz

@dataclass
class INRConfig:
    # Path to the input Point Cloud to reconstruct from. Can be either a single path, or a path that matches to all PCs to 
    # reconstruct from using UNIX stype pathname expansion using *. E.g. 
    # input_pc_path : str = '/media/chamin8TB/MultiresUnorientedNF/data/deep_geometric_prior_data/scans/gargoyle.ply' # or
    # input_pc_path : str = '/media/chamin8TB/MultiresUnorientedNF/data/deep_geometric_prior_data/scans/*.ply'
    input_pc_path : str = '/media/chamin8TB/MultiresUnorientedNF/data/deep_geometric_prior_data/scans/gargoyle.ply'
    shape_name : str = 'gargoyle' # Name of the shape
    dataset_name : str = 'SRB' # Name of the dataset
    # Where to put logs, output and saves
    log_dir: str = '/media/chamin8TB/OG-INR/MultiresUnorientedNF/octree/log/octree_labels2'

    # depth of the octree
    octree_depth: int = 7
    # Path to the serialised, lablled octree. often is "{log_dir}/{shape_name}_depth_{depth}.pkl"
    octree_path: str = ''
    # Path to the saved scaling of the point cloud that the octree code used. Often is "{log_dir}/scaling.npz".
    # If empty, scale to the 
    scaling_path: str = '' 

    # Which INR to use: siren | nglod
    inr_type: str = 'siren'
    hidden_dim: int = 128
    batch_size: int = 40000
    num_iters: int = 1000
    recon_radius : float = 1.1 # After normalising the PC to the unit sphere, what radius cube to do the reconstruction on
    is2D : bool = False # whether to build and octree in 2D or 3D, not currently implemented
    grid_res: int = 256
    lr: float = 1e-3
    gpu_idx: int = 0

    seed: int = 0
    save_vis: bool = True # whether to save a mesh of the current reconstruction
    use_wandb : bool = False # Whether to log using wandb

    notes : str = "nothing much" # Notes to log
    extra_tags : List[str] = field(default_factory=lambda : []) # Extra tags to add
    extra : dict = field(default_factory=lambda : DotDict()) # A place to add extra values

    def __post_init__(self):
        self.shape_log_dir = os.path.join(self.log_dir, self.shape_name)
        if self.octree_path == "":
            self.octree_path = f"{self.shape_log_dir}/{self.shape_name}_depth_{self.octree_depth}.pkl"
        self.num_iters += 1


###########################################################
### Set up args, logging and code/args backup
###########################################################

# parse the cfg
cfg = pyrallis.parse(config_class=INRConfig)
assert cfg.recon_radius >= 1.0, f"recon_radius should be >= 1, but have value {cfg.recon_radius:.5f}"

tags = [cfg.dataset_name, cfg.shape_name, "2D" if cfg.is2D else "3D", cfg.inr_type, *cfg.extra_tags]

# after all changes to the cfg
if cfg.use_wandb:
    import wandb
    wandb.init(
        project = f"OG-INR-SDF_optimisation_{cfg.dataset_name}_dataset",
        notes = cfg.notes,
        tags = tags,
        config = dataclasses.asdict(cfg)
    )

# make the log directory
os.makedirs(cfg.shape_log_dir, exist_ok=True)

# set print 
log_filepath = os.path.join(cfg.shape_log_dir, 'inr_out.log')
print = make_print_also_log(log_filepath) # make print also save to log file
print(cfg)
# print(dataclasses.asdict(cfg))
print("Tags:", tags)

# def log_dict(cost, depth, ot, other={}):
#     return {"cost": cost, "depth": depth, "numSurfaceLeaves": ot.numSurfaceLeaves, "numNonSurfaceLeaves": ot.numNonSurfaceLeaves, 
#                             "numInsideLeaves": ot.labels.sum(), "numOutsideLeaves": (ot.labels==0).sum(), 
#                             "Inside%": ot.labels.mean(), "Outside%": (ot.labels==0).mean(), **other}


os.system('cp %s %s' % (__file__, cfg.shape_log_dir))  # backup the current training file
os.system('cp %s %s' % ('surf_recon_dataset.py', cfg.shape_log_dir))  # backup the default args file
os.system('cp %s %s' % ('../models/NFModel.py', cfg.shape_log_dir))  # backup the models files
os.system('cp %s %s' % ('../models/losses.py', cfg.shape_log_dir))  # backup the layers files


###########################################################
### Set up dataset and model
###########################################################
# Seeds
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('before train data'); t0 = time.time()

# shape_name, pcd_path, octree_pkl_path, octree_depth, scaling_path, batch_size, grid_res=128, grid_radius=1.1)
train_set = shape3D_dataset.ReconDataset(cfg.shape_name, cfg.input_pc_path, cfg.octree_path, cfg.octree_depth, cfg.scaling_path,
                                          cfg.batch_size, cfg.grid_res, grid_radius=cfg.recon_radius, dataset_length=cfg.num_iters)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4,
                                               pin_memory=True)
num_batches = len(train_dataloader)
# grid = train_set.grid
grid=None
print('Loading dataset took {:.3f}s'.format(time.time()-t0)); t0 = time.time()
print('before model')

device = torch.device("cuda:" + str(cfg.gpu_idx) if (torch.cuda.is_available()) else "cpu")
# model = NeuralFieldModel("nglod", grid, dim_in=3, dim_out=1, args=args).to(device)
# model = NeuralFieldModel("siren", grid, dim_in=3, dim_out=1, args=args).to(device)
model = NeuralFieldModel(cfg.inr_type, grid, dim_in=3, dim_out=1, args=cfg).to(device)
n_parameters = count_parameters(model)
print(f"Number of parameters in the current model:{n_parameters}, model_type: {cfg.inr_type}")

optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
# criterion = OGINRLoss(weights=cfg.loss_weights, loss_type=args.loss_type, dataset = train_set)
criterion = OGINRLoss()

print('Loading model and loss took {:.3f}s'.format(time.time()-t0)); t0 = time.time()
time_to_remove = 0

# mnfld, dist, normal, eik, dist2/label
epoch = 0
term_weightings = {'zls_term': 3e3, 'eikonal_term': 5e1, 'approx_sdf_term': 5e4, 'sign_term': 1e3}
for batch_idx, data in enumerate(train_dataloader):
    if batch_idx > 100:
        # criterion.weights = [3e3, 1e4, 5e1, 1e1]
        term_weightings = {'zls_term': 3e3, 'eikonal_term': 5e1, 'approx_sdf_term': 1e4, 'sign_term': 1e2}
    if batch_idx > 200:
        for g in optimizer.param_groups:
            g['lr'] = 5e-4
        # criterion.weights = [3e3, 5e2, 5e1, 1e1]
        term_weightings = {'zls_term': 3e3, 'eikonal_term': 5e1, 'approx_sdf_term': 5e2, 'sign_term': 1e2}
    if batch_idx > 400:
        for g in optimizer.param_groups:
            g['lr'] = 5e-5
        # criterion.weights = [3e3, 1e2, 5e1, 1e-1]
        term_weightings = {'zls_term': 3e3, 'eikonal_term': 5e1, 'approx_sdf_term': 1e2, 'sign_term': 1e0}
        
        
    # save model before update
    if batch_idx % 1000 == 0 :
        t1 = time.time()
        save_name = f'model_{batch_idx}.pth'
        print(f"saving model to file : {save_name}")
        torch.save(model.state_dict(), os.path.join(cfg.shape_log_dir, save_name))
        time_to_remove += time.time() - t1

    model.zero_grad()
    model.train()


    data['pcd_points'] = data['pcd_points'].to(device)
    data['domain_points'] = data['domain_points'].to(device)
    data['domain_approx_df'] = data['domain_approx_df'].to(device)
    data['domain_approx_sdf'] = data['domain_approx_sdf'].to(device)
    data['nonsurf_leaf_samples'] = data['nonsurf_leaf_samples'].to(device)
    data['nonsurf_leaf_sample_signs'] = data['nonsurf_leaf_sample_signs'].to(device) 
    data['surf_leaf_samples'] = data['surf_leaf_samples'].to(device)

    start_time = time.time()
    ctx_dict = {}
    ctx_dict['i'] = batch_idx
    # second argument is whether to calculate normals or not
    input_dict = {'domain': (data['domain_points'], True), 'pcd': (data['pcd_points'], True), 
                'surf_leaf': (data['surf_leaf_samples'], False), 'nonsurf_leaf': (data['nonsurf_leaf_samples'], False)}
    output_pred = model(input_dict, ctx_dict=ctx_dict)
    loss_dict = criterion(output_pred, data, term_weightings=term_weightings)
    
    # Output training stats
    vis_every = 100
    # cfg.save_vis = True
    if batch_idx % vis_every == 0:
        t1 = time.time()
        print("#######################################################################################")
        if cfg.save_vis:
            dir_name = cfg.shape_log_dir
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            full_vises_dirname = f"{dir_name}/vises"
            if not os.path.exists(full_vises_dirname):
                os.makedirs(full_vises_dirname)
            try:
                vis_path = '{}/vises/{}_{}.ply'.format(dir_name,cfg.shape_name, batch_idx)
                implicit_func = lambda points : model.compute_full_grid(points.to(device), process_size=1000000)
                mesh = implicitFunc2mesh(train_set.grid_obj, implicit_func, train_set.scale_obj.unscale_points, chunk_size = 100000, use_tqdm=True)
                mesh.export(vis_path)
            except:
                print(traceback.format_exc())
                print('Could not generate mesh')
                print()
        time_to_remove += time.time() - t1

    if batch_idx % 25 == 0:
        t1 = time.time()
        # weights = criterion.weights
        lr = torch.tensor(optimizer.param_groups[0]['lr'])
        print("iter: {}, weights: {}, lr: {:.2e}".format(batch_idx, term_weightings, lr))
        batch_size = 1
        keys = [key for key in loss_dict.keys() if ('term' in key and 'weighted' not in key)]
        curr_idx, all_idx, curr_perc = batch_idx*batch_size, len(train_set), 100. * batch_idx / len(train_dataloader)
        print(f"Epoch: {epoch} [{curr_idx:4d}/{all_idx} ({curr_perc:.0f}%)]" + 
              f" Loss: {loss_dict['loss']:.5f} = " + 
              " + ".join([f"weighted_{key}: {loss_dict['weighted_'+key]:.5f}" for key in keys]))
        print(f"Epoch: {epoch} [{curr_idx:4d}/{all_idx} ({curr_perc:.0f}%)]" + 
              f" Loss: {loss_dict['loss']:.5f} = " + 
              " + ".join([f"{key}: {loss_dict[key]:.5f}" for key in keys]))
        time_to_remove += time.time() - t1

    loss = loss_dict["loss"]
    loss.backward()
    optimizer.step()
    # scheduler.step()

    # save last model
    if batch_idx == num_batches - 1 :
        t1 = time.time()
        print("saving model to file :{}".format('model_%d.pth' % (batch_idx)))
        torch.save(model.state_dict(),
                    os.path.join(cfg.shape_log_dir, 'model_%d.pth' % (batch_idx)))
        time_to_remove += time.time() - t1