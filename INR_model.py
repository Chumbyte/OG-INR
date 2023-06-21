import os,sys,time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import grad

def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True)[0]
    return points_grad

def get_main_module(nf, model_type, grid, args):
    if model_type == 'nglod':
        print('Loading NGLOD model')
        return OctreeSDF(nf.dim_in, args)
    else:
        assert model_type == 'siren'
        print('Loading SIREN model')
        return SIREN(nf.dim_in, args=args)

class NeuralFieldModel(nn.Module):
    def __init__(self, model_type, grid, dim_in=3, dim_out=1, args={}):
        # grid is a Grid() object which gives more info on the grid details
        super().__init__()
        self.model_type = model_type
        self.grid = grid
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.main_module = get_main_module(self, self.model_type, grid, args)
    
    def forward(self, input_dict, ctx_dict=None):
        # input_dict[input_name:str] = (input_points: Tensor(1, n, in_dim), return_normals: bool)
        out_dict = {}
        for input_name in input_dict:
            input_points, return_normals = input_dict[input_name]
            assert len(input_points.shape) == 3, input_points.shape
            assert input_points.shape[-1] == self.dim_in, (input_points.shape, self.dim_in)
            if return_normals:
                input_points.requires_grad_()
            output = self.main_module(input_points[0], return_normals=return_normals, ctx_dict=ctx_dict)
            out_dict[input_name] = output
                
        return out_dict
    
    def compute_full_grid(self, all_grid_points, process_size=10000):
        # t0 = time.time()
        all_grid_points = all_grid_points.reshape(-1, self.dim_in)
        num_grid_points = all_grid_points.shape[0]
        
        grid_sdfs = []
        with torch.no_grad():
            for i in range(math.ceil(num_grid_points / process_size)):
                current_batch = all_grid_points[i*process_size:(i+1)*process_size]
                res_dict = self.main_module(current_batch, return_normals=False, ctx_dict=None)
                grid_sdfs.append(res_dict['pred'].reshape(-1))
        grid_sdfs = torch.cat(grid_sdfs)
        # print("compute_full_grid ended, ({:.3f})s".format(time.time()-t0))
        return grid_sdfs

####################################################################################################
###################################  SIREN architecture ############################################
####################################################################################################

# From the SIREN official code
class SIREN(nn.Module):

    def __init__(self, in_features, num_hidden_layers=4, hidden_features=256,
                 outermost_linear=True, nonlinearity='sine', init_type='siren',args=None):
        super().__init__()
        print("decoder initialising with {} and {}".format(nonlinearity, init_type))


        nl_dict = {'sine': Sine(), 'relu': nn.ReLU(inplace=True), 'softplus': nn.Softplus(beta=100),
                    'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}
        nl = nl_dict[nonlinearity]

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, 1)))
        else:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, 1), nl))

        self.net = nn.Sequential(*self.net)

        # self.init_type = init_type
        # if init_type == 'siren':
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)


    def forward(self, x, return_normals=True, ctx_dict=None):
        # x: (N,3)
        return_dict = {}
        output = self.net(x) # (N, 1)

        return_dict['pred'] = output.unsqueeze(0) # (1,N,1)
        # return_normals = True
        if return_normals:
            return_dict['grad'] = gradient(x, output).unsqueeze(0) # (1,N,input_dim)

        return return_dict

# SIREN's activation
class Sine(nn.Module):
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

# SIREN's initialization
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See SIREN paper supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

####################################################################################################
###################################  NGLOD architecture ############################################
####################################################################################################

# Adapted from the NGLOD code. Note that multi-resolution wasn't implemented at the time.
class FeatureVolume(nn.Module):
    def __init__(self, input_dim, fdim, fsize, domain_range=1.2):
        super().__init__()
        self.input_dim = input_dim
        self.fsize = fsize
        self.fdim = fdim
        # f_size+1 so that you have values at the vertices of a 3D grid of size f_size x f_size x f_size
        if self.input_dim == 3:
            self.fm = nn.Parameter(torch.randn(1, fdim, fsize+1, fsize+1, fsize+1) * 0.01)
        else:
            self.fm = nn.Parameter(torch.randn(1, fdim, fsize+1, fsize+1) * 0.01)
        self.sparse = None
        self.domain_range = domain_range

    def forward(self, x):
        # x is (N,input_dim)
        assert len(x.shape) == 2, x.shape
        N = x.shape[0]
        assert x.shape[1] == self.input_dim
        x = x / self.domain_range # transform domain to [-1,1]^(input_dim) from [-dr,dr]^(input_dim)

        if self.input_dim == 3:
            # Grid sample takes in (N,C,D_in,H_in,W_in), (N,D_out,H_out,W_out,2) and returns (N,C,D_out,H_out,W_out)
            # we will make num points N on the D_out rather than batch size (N) dimension
            sample_coords = x.reshape(1, N, 1, 1, 3)   
            # sample = F.grid_sample(self.fm, sample_coords, 
            sample = grid_sample_3d(self.fm, sample_coords, 
                                   align_corners=True, padding_mode='border')[0,:,:,0,0].transpose(0,1)
            # Align corners is important
            # input coords normalised to [-1,1]^3. N coords results in a (1,N,1,1,3) tensor
            # self.fm is the grid of fdim sized latent vectors, a (1,fdim,fs+1,fs+1,fs+1) tensor
            # returns a (1,fdim,N,1,1) tensor that is changed to (N,fdim)
        else:
            assert self.input_dim == 2, self.input_dim
            # # # Grid sample takes in (N,C,H_in,W_in), (N,H_out,W_out,2) and returns (N,C,H_out,W_out)
            # # # we will make num points N on the H_out rather than batch size (N) dimension
            # # sample_coords = x.reshape(1, N, 1, 2)
            # # # sample = F.grid_sample(self.fm, sample_coords, 
            # # sample = grid_sample_2d(self.fm, sample_coords, 
            # #                        align_corners=True, padding_mode='border')[0,:,:,0].transpose(0,1)
            # # # Align corners is important
            # # # input coords normalised to [-1,1]^3. N coords results in a (1,N,1,2) tensor
            # # # self.fm is the grid of fdim sized latent vectors, a (1,fdim,fs+1,fs+1) tensor
            # # # returns a (1,fdim,N,1) tensor that is changed to (N,fdim)

            # grid_vals = self.fm[0] # (C,H,W)
            # coords = x # (N, 2)
            # sample = interpolate_grid_2d(grid_vals, coords).transpose(0,1) # (N, fdim)

            pass
        
        return sample

class OctreeSDF(nn.Module):
    def __init__(self, input_dim, args, init=None):
        super().__init__()

        self.input_dim = input_dim # 2 or 3
        self.fdim = 32
        # self.fsize = 4
        self.hidden_dim = 128
        self.pos_invariant = False
        # self.joint_decoder = False
        self.joint_decoder = True
        self.num_lods = 1
        # self.num_lods = 3
        self.interpolate = None
        # self.base_lod = 2
        self.base_lod = 7
        self.lod = None # No default lod to eval at

        self.features = nn.ModuleList([])
        for i in range(self.num_lods):
            self.features.append(FeatureVolume(input_dim, self.fdim, (2**(i+self.base_lod))))

        self.louts = nn.ModuleList([])

        self.sdf_input_dim = self.fdim
        if not self.pos_invariant:
            self.sdf_input_dim += self.input_dim

        self.num_decoder = 1 if self.joint_decoder else self.num_lods 

        for i in range(self.num_decoder):
            self.louts.append(
                nn.Sequential(
                    nn.Linear(self.sdf_input_dim, self.hidden_dim, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1, bias=True),
                )
            )
        
    def encode(self, x):
        # Disable encoding
        return x
    
    def forward(self, x, return_normals=True, ctx_dict=None):
        return self.sdf(x, return_normals=return_normals)

    def sdf(self, x, lod=None, return_lst=False, return_normals=True):
        # x is (N,input_dim)

        return_dict = {}

        if lod is None:
            lod = self.lod
        
        # Query
        l = []
        samples = []

        for i in range(self.num_lods):
            
            # Query features
            sample = self.features[i](x)
            samples.append(sample)
            
            # Sum queried features
            if i > 0:
                samples[i] += samples[i-1]
            
            # Concatenate xyz
            ex_sample = samples[i]
            if not self.pos_invariant:
                ex_sample = torch.cat([x, ex_sample], dim=-1)

            if self.num_decoder == 1:
                prev_decoder = self.louts[0]
                curr_decoder = self.louts[0]
            else:
                prev_decoder = self.louts[i-1]
                curr_decoder = self.louts[i]
            

            # Interpolation mode (only for prediction mode, i.e. lod is set)
            if self.interpolate is not None and lod is not None:
                d = curr_decoder(ex_sample) # (N, 1)
                
                if i == len(self.louts) - 1:
                    return d

                if lod+1 == i:
                    _ex_sample = samples[i-1]
                    if not self.pos_invariant:
                        _ex_sample = torch.cat([x, _ex_sample], dim=-1)
                    _d = prev_decoder(_ex_sample)

                    return (1.0 - self.interpolate) * _d + self.interpolate * d
            
            # Get distance
            else: 
                d = curr_decoder(ex_sample) # (N, 1)

                # Return distance if in prediction mode
                if lod is not None and lod == i:
                    return d

                l.append(d)
        if self.training:
            self.loss_preds = l

        # if return_lst:
        #     return l
        # else:
        #     return l[-1]
        


        return_dict['pred'] = l[-1].reshape(1,-1,1) # (1,N,1)
        # return_normals = True
        if return_normals:
            return_dict['grad'] = gradient(x, l[-1]).unsqueeze(0) # (1,N,input_dim)

        return return_dict


# https://github.com/pytorch/pytorch/issues/34704
def grid_sample_2d(image, optical, **kwargs):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)         # (1,N,1)
    ne = (ix    - ix_sw) * (iy_sw - iy)         # (1,N,1)
    sw = (ix_ne - ix)    * (iy    - iy_ne)      # (1,N,1)
    se = (ix    - ix_nw) * (iy    - iy_nw)      # (1,N,1)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def grid_sample_3d(image, optical, **kwargs):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)


    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val


############################################################################################################################################################

