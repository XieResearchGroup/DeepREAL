from rdkit import Chem
import torch
from torch_geometric.data import Data
import numpy as np

# allowable node and edge features
# allowable_features = {
#     'possible_atomic_num_list' : list(range(1, 119)),
#     'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
#     'possible_chirality_list' : [
#         Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
#         Chem.rdchem.ChiralType.CHI_OTHER
#     ],
#     'possible_hybridization_list' : [
#         Chem.rdchem.HybridizationType.S,
#         Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
#         Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
#         Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
#     ],
#     'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
#     'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
#     'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'possible_bonds' : [
#         Chem.rdchem.BondType.SINGLE,
#         Chem.rdchem.BondType.DOUBLE,
#         Chem.rdchem.BondType.TRIPLE,
#         Chem.rdchem.BondType.AROMATIC
#     ],
#     'possible_bond_dirs' : [ # only for double bond stereo information
#         Chem.rdchem.BondDir.NONE,
#         Chem.rdchem.BondDir.ENDUPRIGHT,
#         Chem.rdchem.BondDir.ENDDOWNRIGHT
#     ]
# }

allowable_features = { # from Yang
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_hybridization_list": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_aromatic_list": [True, False],
    "possible_bond_dirs": [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}

# def mol_to_graph_data_obj_simple(mol):
#     """
#     Converts rdkit mol object to graph Data object required by the pytorch
#     geometric package. NB: Uses simplified atom and bond features, and represent
#     as indices
#     :param mol: rdkit mol object
#     :return: graph data object with the attributes: x, edge_index, edge_attr
#     """
#     # atoms
#     num_atom_features = 2   # atom type,  chirality tag
#     atom_features_list = []
#     for atom in mol.GetAtoms():
#         atom_feature = [allowable_features['possible_atomic_num_list'].index(
#             atom.GetAtomicNum())] + [allowable_features[
#             'possible_chirality_list'].index(atom.GetChiralTag())]
#         atom_features_list.append(atom_feature)
#     x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
#
#     # bonds
#     num_bond_features = 2   # bond type, bond direction
#     if len(mol.GetBonds()) > 0: # mol has bonds
#         edges_list = []
#         edge_features_list = []
#         for bond in mol.GetBonds():
#             i = bond.GetBeginAtomIdx()
#             j = bond.GetEndAtomIdx()
#             edge_feature = [allowable_features['possible_bonds'].index(
#                 bond.GetBondType())] + [allowable_features[
#                                             'possible_bond_dirs'].index(
#                 bond.GetBondDir())]
#             edges_list.append((i, j))
#             edge_features_list.append(edge_feature)
#             edges_list.append((j, i))
#             edge_features_list.append(edge_feature)
#
#         # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
#         edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
#
#         # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
#         edge_attr = torch.tensor(np.array(edge_features_list),
#                                  dtype=torch.long)
#     else:   # mol has no bonds
#         edge_index = torch.empty((2, 0), dtype=torch.long)
#         edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
#
#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
#
#     return data

def mol_to_graph_data_obj_simple(mol): # from Yang
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    # num_atom_features = 6  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = (
            [allowable_features["possible_atomic_num_list"].index(atom.GetAtomicNum())]
            + [allowable_features["possible_degree_list"].index(atom.GetDegree())]
            + [allowable_features["possible_formal_charge_list"].index(atom.GetFormalCharge())]
            + [
                allowable_features["possible_hybridization_list"].index(
                    atom.GetHybridization()
                )
            ]
            + [allowable_features["possible_aromatic_list"].index(atom.GetIsAromatic())]
            + [allowable_features["possible_chirality_list"].index(atom.GetChiralTag())]
        )
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                allowable_features["possible_bonds"].index(bond.GetBondType())
            ] + [allowable_features["possible_bond_dirs"].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data