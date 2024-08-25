def global_parameters_set():
    config = {}
    config['dataset_name'] = 'BIOSNAP'

    config['batch_size'] = 128
    config['train_epoch'] = 100
    config['workers'] = 0
    config['learning_rate'] = 1e-3
    config['dropout_rate'] = 0.2

    config['drug_length_max'] = 100

    config['residue_num_max'] = 512
    config['contact_threshold'] = 8 * 1e-10
    config['residue_features_esmv2_size'] = '8M'
    config['contact_map_esmv2_size'] = '8M'

    config['drug_sequence_length_max'] = 100
    config['protein_vocab_size'] = 16693
    config['emb_size'] = 384
    config['protein_length_max'] = 450

    return config
