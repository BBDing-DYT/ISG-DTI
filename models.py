"""该类主要存放网络模型"""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from model_components import WordEmbeddingAndPosEncoding, SAPBlock, NormalGAT, \
    SimpleCrossTransformerEncoderBlock


class CrossSequenceModel(nn.Module):
    def __init__(self, use_softmax_cross_entropy_loss=False, **config):
        super().__init__()
        self.use_softmax_cross_entropy_loss = use_softmax_cross_entropy_loss
        self.drug_length_max = config['drug_length_max']
        self.protein_vocab_size = config['protein_vocab_size']
        self.protein_length_max = config['protein_length_max']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate']
        self.batch_size = config['batch_size']

        self.protein_embedding_layer = WordEmbeddingAndPosEncoding(self.protein_vocab_size, self.emb_size, self.protein_length_max,
                                                                   self.dropout_rate)
        self.drug_preprocess_FFN = nn.Sequential(
            nn.LazyLinear(128, True),
            nn.LayerNorm(128)
        )
        self.protein_preprocess_conv1d = nn.Sequential(
            nn.Conv1d(384, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.protein_preprocess_layer_norm =nn.LayerNorm(128)

        self.drug_gnn = NormalGAT(3, 128, 4, True)
        self.cross_transformer_encoder_block = SimpleCrossTransformerEncoderBlock(4,
                                                                            128, 256,
                                                                            128, 256,
                                                                            128, self.dropout_rate, True)

        self.drug_SSA = SAPBlock(64, 1)
        self.protein_SSA = SAPBlock(64, 1)

        self.last_fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.LazyLinear(64),
            nn.GELU(approximate='tanh'),
        )
        if not self.use_softmax_cross_entropy_loss:
            self.last_fc.append(nn.LazyLinear(1))
        else:
            self.last_fc.append(nn.LazyLinear(2))

    def forward(self, atom_num, drug_graph, protein_sequence_length, protein_one_hot_index_list):
        protein_features = self.protein_embedding_layer(protein_one_hot_index_list)
        drug_features, edge_index, batch = drug_graph.x, drug_graph.edge_index, drug_graph.batch
        node_num = torch.bincount(batch)

        drug_features = self.drug_preprocess_FFN(drug_features)
        protein_features = protein_features.permute(0, 2, 1)
        protein_features = self.protein_preprocess_conv1d(protein_features)
        protein_features = protein_features.permute(0, 2, 1)
        protein_features = self.protein_preprocess_layer_norm(protein_features)

        drug_features = self.drug_gnn(drug_features, edge_index)

        drug_features = torch.split(drug_features, node_num.tolist())
        drug_features = pad_sequence(list(drug_features), batch_first=True, padding_value=0)

        if drug_features.size(1) < self.drug_length_max:
            padding = (torch.zeros(drug_features.size(0), self.drug_length_max - drug_features.size(1), drug_features.size(2))
                       .to(drug_features.device))
            drug_features = torch.cat([drug_features, padding], dim=1)
        drug_features, protein_features = self.cross_transformer_encoder_block(drug_features, atom_num, protein_features, protein_sequence_length)

        drug_features = self.drug_SSA(drug_features, atom_num).squeeze()
        protein_features = self.protein_SSA(protein_features, protein_sequence_length).squeeze()

        x = torch.cat((drug_features, protein_features), dim=-1)
        x = self.last_fc(x)

        return x


class CrossGraphModel(nn.Module):
    def __init__(self, use_softmax_cross_entropy_loss=False, **config):
        super().__init__()
        self.drug_length_max = config['drug_length_max']
        self.residue_num_max = config['residue_num_max']
        self.residue_features_esmv2_size = config['residue_features_esmv2_size']

        self.use_softmax_cross_entropy_loss = use_softmax_cross_entropy_loss

        self.dropout_rate = config['dropout_rate']

        self.drug_preprocess = nn.Sequential(
            nn.LazyLinear(128),
            # nn.GELU(approximate='tanh')
        )
        if self.residue_features_esmv2_size == '650M':
            self.protein_preprocess = nn.Sequential(
                nn.LazyLinear(512),
                nn.LazyLinear(256),
                # nn.GELU(approximate='tanh')
            )
        else:
            self.protein_preprocess = nn.Sequential(
                nn.LazyLinear(256),
                # nn.GELU(approximate='tanh')
            )

        self.drug_gnn = NormalGAT(3, 128, 4, True)
        self.protein_gnn = NormalGAT(3, 256, 4, True)

        self.cross_transformer_encoder_block = SimpleCrossTransformerEncoderBlock(4,
                                                                            128, 64,
                                                                            256, 128,
                                                                            16, self.dropout_rate, True)

        # self.drug_pool = global_max_pool
        # self.protein_pool = global_max_pool
        self.drug_pool = SAPBlock(128, 1)
        self.protein_pool = SAPBlock(256, 1)

        self.predict_layer = nn.Sequential(
            nn.LazyLinear(64),
            # nn.GELU(approximate='tanh')
        )
        if not self.use_softmax_cross_entropy_loss:
            self.predict_layer.append(nn.LazyLinear(1))
        else:
            self.predict_layer.append(nn.LazyLinear(2))

    def forward(self, atom_num, drug_graph, residue_num, contact_map):
        drug_features, drug_edge_index, drug_batch = drug_graph.x, drug_graph.edge_index, drug_graph.batch
        drug_node_num = torch.bincount(drug_batch)
        residue_features, protein_edge_index, protein_batch = contact_map.x, contact_map.edge_index, contact_map.batch
        protein_node_num = torch.bincount(protein_batch)

        drug_features = self.drug_preprocess(drug_features)
        residue_features = self.protein_preprocess(residue_features)

        drug_features = self.drug_gnn(drug_features, drug_edge_index)
        residue_features = self.protein_gnn(residue_features, protein_edge_index)


        drug_features = torch.split(drug_features, drug_node_num.tolist())
        drug_features = pad_sequence(list(drug_features), batch_first=True, padding_value=0)
        if drug_features.size(1) < self.drug_length_max:
            padding = (torch.zeros(drug_features.size(0), self.drug_length_max - drug_features.size(1), drug_features.size(2))
                       .to(drug_features.device))
            drug_features = torch.cat([drug_features, padding], dim=1)
        residue_features = torch.split(residue_features, protein_node_num.tolist())
        residue_features = pad_sequence(list(residue_features), batch_first=True, padding_value=0)
        if residue_features.size(1) < self.residue_num_max:
            padding = (torch.zeros(residue_features.size(0), self.residue_num_max - residue_features.size(1), residue_features.size(2))
                       .to(residue_features.device))
            residue_features = torch.cat([residue_features, padding], dim=1)

        drug_features, residue_features = self.cross_transformer_encoder_block(drug_features, atom_num, residue_features, residue_num)

        drug_feature = self.drug_pool(drug_features, atom_num).squeeze()
        protein_feature = self.protein_pool(residue_features, residue_num).squeeze()

        x = torch.cat((drug_feature, protein_feature), dim=-1)
        x = self.predict_layer(x)

        return x
