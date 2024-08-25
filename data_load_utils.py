import codecs
import hashlib

import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from subword_nmt.apply_bpe import BPE
from torch.utils import data
from torch_geometric import data as DATA

from config import global_parameters_set


protein_bpe = BPE(codecs.open('./ESPF/protein_codes_uniprot.txt'), merges=-1, separator='')
protein_subword = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')['index'].values
protein_subword2index = dict(zip(protein_subword, range(1, len(protein_subword) + 1)))
global_parameters = global_parameters_set()


class DTIDataset(data.Dataset):
    def __init__(self, raw_data_df):
        processed_df = raw_data_df[raw_data_df.iloc[:, 0].apply(lambda x: Chem.MolFromSmiles(x) is not None)]  # 某些SMILES序列是错的，无法解析，删除这些数据
        processed_df = processed_df.reset_index(drop=True)  # 重置行索引
        self.df = processed_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        drug_sequence = self.df.iloc[index]['SMILES']
        atom_num, drug_features, edge_index = _smiles_to_graph(drug_sequence)
        atom_num = torch.tensor(atom_num)
        drug_graph = DATA.Data(x=torch.Tensor(drug_features),
                               edge_index=torch.LongTensor(edge_index).transpose(1, 0))

        protein_sequence = self.df.iloc[index]['Target Sequence']

        protein_sequence_length, protein_one_hot_index_list = _protein2emb_encoder(protein_sequence)
        protein_sequence_length = torch.tensor(protein_sequence_length)
        protein_one_hot_index_list = torch.from_numpy(protein_one_hot_index_list)

        label = self.df.iloc[index]['Label']
        label = torch.tensor(label)

        return atom_num, drug_graph, protein_sequence_length, protein_one_hot_index_list, label


def drug_graph_and_protein_sequence_collate_fn(batch):
    atom_nums = [item[0] for item in batch]
    drug_graphs = [item[1] for item in batch]
    protein_sequence_lengths = [item[2] for item in batch]
    protein_one_hot_index_lists = [item[3] for item in batch]
    labels = [item[4] for item in batch]
    # use torch.stack to collate PyTorch tensors
    batched_atom_nums = torch.stack(atom_nums)
    # use torch_geometric.data.Batch to collate graphs
    batched_drug_graphs = DATA.Batch.from_data_list(drug_graphs)
    batched_protein_sequence_lengths = torch.stack(protein_sequence_lengths)
    batched_protein_one_hot_index_lists = torch.stack(protein_one_hot_index_lists)
    batched_labels = torch.stack(labels)
    # return a dictionary of batched data
    return batched_atom_nums, batched_drug_graphs, batched_protein_sequence_lengths, batched_protein_one_hot_index_lists, batched_labels


class TwoGraphDTIDataset(data.Dataset):
    def __init__(self, raw_data_df):
        processed_df = raw_data_df[raw_data_df.iloc[:, 0].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
        processed_df = processed_df.reset_index(drop=True)
        self.df = processed_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        drug_sequence = self.df.iloc[index]['SMILES']
        atom_num, drug_features, drug_edge_index = _smiles_to_graph(drug_sequence)
        atom_num = torch.tensor(atom_num)
        drug_graph = DATA.Data(x=torch.Tensor(drug_features),
                               edge_index=torch.LongTensor(drug_edge_index).transpose(1, 0))

        protein_sequence = self.df.iloc[index]['Target Sequence']
        residue_num, residue_features, residue_edge_index = _fasta2graph(protein_sequence)
        residue_num = torch.tensor(residue_num)
        contact_graph = DATA.Data(x=torch.Tensor(residue_features),
                                  edge_index=torch.LongTensor(residue_edge_index).transpose(1, 0))


        label = self.df.iloc[index]['Label']
        label = torch.tensor(label)

        return atom_num, drug_graph, residue_num, contact_graph, label


def drug_graph_and_contact_graph_collate_fn(batch):
    atom_nums = [item[0] for item in batch]
    drug_graphs = [item[1] for item in batch]
    residue_nums = [item[2] for item in batch]
    contact_graphs = [item[3] for item in batch]
    labels = [item[4] for item in batch]
    # use torch.stack to collate PyTorch tensors
    batched_atom_nums = torch.stack(atom_nums)
    # use torch_geometric.data.Batch to collate graphs
    batched_drug_graphs = DATA.Batch.from_data_list(drug_graphs)
    batched_residue_nums = torch.stack(residue_nums)
    batched_contact_graphs = DATA.Batch.from_data_list(contact_graphs)
    batched_labels = torch.stack(labels)
    # return a dictionary of batched data
    return batched_atom_nums, batched_drug_graphs, batched_residue_nums, batched_contact_graphs, batched_labels


def _key_node_extract(node_num_max, features, graph):
    current_node_num = graph.number_of_nodes()
    flag = False
    nodes_to_remove = []

    while current_node_num > node_num_max:
        flag = True
        min_degree_node = min(graph, key=graph.degree)
        nodes_to_remove.append(min_degree_node)
        graph.remove_node(min_degree_node)
        current_node_num -= 1

    if flag:
        features = np.delete(features, nodes_to_remove, axis=0)
        mapping = {node: i for i, node in enumerate(graph.nodes)}
        graph = nx.relabel_nodes(graph, mapping)

    return current_node_num, features, graph


def _protein2emb_encoder(x):
    protein_length_max = global_parameters['protein_length_max']
    sub_word_list = protein_bpe.process_line(x).split()
    try:
        one_hot_index_list = np.asarray([protein_subword2index[i] for i in sub_word_list])
    except:
        one_hot_index_list = np.array([0])

    protein_sequence_length = len(one_hot_index_list)

    if protein_sequence_length < protein_length_max:
        final_one_hot_index_list = np.pad(one_hot_index_list, (0, protein_length_max - protein_sequence_length), 'constant', constant_values=0)
    else:
        final_one_hot_index_list = one_hot_index_list[:protein_length_max]
        protein_sequence_length = protein_length_max

    return protein_sequence_length, final_one_hot_index_list


def _atom_features(atom):
    def _one_of_k_encoding(x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def _one_of_k_encoding_unk(x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    return np.array(_one_of_k_encoding_unk(atom.GetSymbol(),
                                           ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    _one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    _one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    _one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def _smiles_to_graph(smiles):
    atom_num_max = global_parameters['drug_length_max']
    mol = Chem.MolFromSmiles(smiles)

    graph = nx.Graph()
    features = []
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx())
        feature = _atom_features(atom)
        features.append(feature / sum(feature))
    features = np.array(features)

    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    actual_atom_num, features, graph = _key_node_extract(atom_num_max, features, graph)

    graph = graph.to_directed()
    edge_index = []
    for node_u, node_v in graph.edges:
        edge_index.append([node_u, node_v])
    if len(edge_index) == 0:
        edge_index = np.empty([0, 2], dtype=np.int64)
    else:
        edge_index = np.array(edge_index)

    return actual_atom_num, features, edge_index


def _process_contact_map(raw_contact_map, contact_threshold):
    processed_contact_map = np.where(raw_contact_map <= contact_threshold, 1, 0)
    np.fill_diagonal(processed_contact_map, 0)
    processed_contact_map = processed_contact_map.astype(np.int64)

    return processed_contact_map


def _fasta2graph(fasta):
    def _hash_cal(str):
        if str is None or str.isspace():
            raise ValueError('str is None Or empty')
        sha224gen = hashlib.sha224()
        sha224gen.update(str.encode())
        sha224code = sha224gen.hexdigest()
        return sha224code

    residue_num_max = global_parameters['residue_num_max']
    contact_threshold = global_parameters['contact_threshold']
    esmv2_size_residue_features = global_parameters['residue_features_esmv2_size']
    esmv2_size_contact_map = global_parameters['contact_map_esmv2_size']
    dataset_name = global_parameters['dataset_name']

    fasta_hash = _hash_cal(fasta)
    residue_features_file_path = './esmv2/datasets/' + dataset_name + '/residue_features_' + esmv2_size_residue_features + '/' + fasta_hash + '.npy'
    contact_map_file_path = './esmv2/datasets/' + dataset_name + '/contanct_map_' + esmv2_size_contact_map + '/' + fasta_hash + '.npy'
    residue_features = np.load(residue_features_file_path)
    contact_map = np.load(contact_map_file_path)
    processed_contact_map = _process_contact_map(contact_map, contact_threshold)

    graph = nx.from_numpy_array(processed_contact_map)
    actual_residue_num, residue_features, graph = _key_node_extract(residue_num_max, residue_features, graph)

    graph = graph.to_directed()
    edge_index = []
    for node_u, node_v in graph.edges:
        edge_index.append([node_u, node_v])
    if len(edge_index) == 0:
        edge_index = np.empty([0, 2], dtype=np.int64)
    else:
        edge_index = np.array(edge_index)

    return actual_residue_num, residue_features, edge_index
