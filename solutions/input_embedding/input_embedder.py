import torch
from config import InputEmbeddingConfig
from feature_extraction.feature_extraction import Batch
from torch import nn
import torch.nn.functional as F

from input_embedding.atom_attention import AtomAttentionEncoder


class InputEmbedder(nn.Module):
    def __init__(self, c_s, c_z, target_feat_dim, rel_feat_dim, config: InputEmbeddingConfig):
        super().__init__()
        self.left_single = nn.Linear(target_feat_dim, c_z, bias=False)
        self.right_single = nn.Linear(target_feat_dim, c_z, bias=False)
        self.position_activations = nn.Linear(rel_feat_dim, c_z, bias=False)
        self.bond_embedding = nn.Linear(1, c_z, bias=False)
        self.single_embedding = nn.Linear(target_feat_dim, c_s, bias=False)
        self.atom_cross_att = AtomAttentionEncoder(c_s, c_z, config.atom_attention_config, use_trunk=False)

        self.r_max = config.r_max
        self.s_max = config.s_max

    def relative_encoding(self, batch: Batch):
        # Implements Algorithm 3 from the paper
        token_features = batch.token_features
        token_index = token_features.token_index
        residue_index = token_features.residue_index
        asym_id = token_features.asym_id
        entity_id = token_features.entity_id
        sym_id = token_features.sym_id

        left_token_index, right_token_index = token_index[...,
                                                          None], token_index[..., None, :]
        left_residue_index, right_residue_index = residue_index[...,
                                                                None], residue_index[..., None, :]
        left_asym_id, right_asym_id = asym_id[..., None], asym_id[..., None, :]
        left_entity_id, right_entity_id = entity_id[...,
                                                    None], entity_id[..., None, :]
        left_sym_id, right_sym_id = sym_id[..., None], sym_id[..., None, :]

        same_chain = left_asym_id == right_asym_id
        same_residue = left_residue_index == right_residue_index
        same_entity = left_entity_id == right_entity_id

        residue_dist = torch.clip(
            left_residue_index-right_residue_index+self.r_max, 0, 2*self.r_max)
        residue_dist[~same_chain] = 2*self.r_max+1

        token_dist = torch.clip(
            left_token_index-right_token_index + self.r_max, 0, 2*self.r_max)
        token_dist[~(same_chain & same_residue)] = 2*self.r_max+1

        chain_dist = torch.clip(left_sym_id-right_sym_id+self.s_max, 0, 2*self.s_max)
        chain_dist[~same_entity] = 2*self.s_max+1

        a_rel_pos = F.one_hot(residue_dist, 2*self.r_max+2)
        a_rel_token = F.one_hot(token_dist, 2*self.r_max+2)
        a_rel_chain = F.one_hot(chain_dist, 2*self.s_max+2)

        p = torch.cat(
            (a_rel_pos, a_rel_token, same_entity[..., None], a_rel_chain), dim=-1)
        rel_feat = p.to(dtype=torch.float32)
        rel_enc = self.position_activations(rel_feat)

        return rel_enc, rel_feat

    def forward(self, batch):
        # Implements Line 1 to Line 5 from Algorithm 1
        target_feat = batch.msa_features.target_feat
        token_act, _ = self.atom_cross_att(batch.ref_struct)
        s_input = torch.cat((target_feat, token_act), dim=-1)

        s_init = self.single_embedding(s_input)
        a = self.left_single(s_input)
        b = self.right_single(s_input)
        z_init = a[..., None, :] + b[..., None, :, :]

        rel_enc, rel_feat = self.relative_encoding(batch)
        z_init += rel_enc
        z_init += self.bond_embedding(batch.contact_matrix)

        return s_input, s_init, z_init, rel_feat
