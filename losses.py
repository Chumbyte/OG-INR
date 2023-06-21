import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def eikonal_loss(domain_grad, pcd_grad=None, eikonal_type='abs'):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1
    # shape is (bs, num_points, dim=3) for both grads

    if domain_grad is not None and pcd_grad is not None:
        all_grads = torch.cat([domain_grad, pcd_grad], dim=-2)
    elif domain_grad is not None:
        all_grads = domain_grad
    elif pcd_grad is not None:
        all_grads = pcd_grad
    
    if eikonal_type == 'abs':
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()

    return eikonal_term

class OGINRLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.term_weightings = {'zls_term': 3e3, 'eikonal_term': 5e1, 'approx_sdf_term': 1e2, 'sign_term': 1e2}

    def forward(self, output_pred, data, term_weightings=None, ctx_dict=None):
        # data: 'pcd_points', 'domain_points', 'domain_approx_df', 'domain_approx_sdf', 'surf_leaf_samples'
        #       'nonsurf_leaf_samples', 'nonsurf_leaf_sample_signs'
        dims = data['domain_points'].shape[-1]
        device = data['domain_points'].device
        if term_weightings is None:
            term_weightings = self.term_weightings

        loss_dict = {}
        loss = 0.0

        #########################################
        # Compute required terms
        #########################################

        domain_pred = output_pred['domain']['pred']
        pcd_pred = output_pred['pcd']['pred']
        domain_grad = output_pred['domain']['grad'].reshape(1,-1,dims)
        pcd_grad = output_pred['pcd']['grad']

        surf_lf_pred = output_pred['surf_leaf']['pred'] # (1,Ns*k,1)
        nonsurf_lf_pred = output_pred['nonsurf_leaf']['pred'] # (1,Nns*k,1)

        # surface / zero-level-set term
        if 'zls_term' in term_weightings:
            loss_dict['zls_term'] = torch.abs(pcd_pred).mean()
            loss_dict['weighted_zls_term'] = loss_dict['zls_term'] * term_weightings['zls_term']
            loss += loss_dict['weighted_zls_term']

        # eikonal term
        if 'eikonal_term' in term_weightings:
            loss_dict['eikonal_term'] = eikonal_loss(domain_grad, pcd_grad=pcd_grad, eikonal_type='abs')
            loss_dict['weighted_eikonal_term'] = loss_dict['eikonal_term'] * term_weightings['eikonal_term']
            loss += loss_dict['weighted_eikonal_term']

        # approximate sdf term: match SDF with sign from octree and dist as dist to input PCD
        if 'approx_sdf_term' in term_weightings:
            loss_dict['approx_sdf_term'] = (data['domain_approx_sdf'].squeeze() - domain_pred.squeeze()).abs().mean()
            loss_dict['weighted_approx_sdf_term'] = loss_dict['approx_sdf_term'] * term_weightings['approx_sdf_term']
            loss += loss_dict['weighted_approx_sdf_term']

        # enforce octree sign for fixed amount of samples in each octree leaf
        if 'sign_term' in term_weightings:
            is_outside = data['nonsurf_leaf_sample_signs'].reshape(-1)==1
            num_lf_samples = data['nonsurf_leaf_samples'].shape[1] + data['surf_leaf_samples'].shape[1]

            sign_term = (nonsurf_lf_pred.reshape(-1)[is_outside]*-1).clamp(min=0).sum()
            sign_term += (nonsurf_lf_pred.reshape(-1)[~is_outside]).clamp(min=0).sum()
            sign_term += surf_lf_pred.reshape(-1).clamp(min=0).sum()
            sign_term /= num_lf_samples
            # import pdb; pdb.set_trace()
            loss_dict['sign_term'] = sign_term
            loss_dict['weighted_sign_term'] = loss_dict['sign_term'] * term_weightings['sign_term']
            loss += loss_dict['weighted_sign_term']
        
        loss_dict['loss'] = loss
        return loss_dict
