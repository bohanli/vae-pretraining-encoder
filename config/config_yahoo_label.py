
params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'batch_size': 32,
    'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'datasets/yahoo_label_data/yahoo.train.txt',
    'val_data': 'datasets/yahoo_label_data/yahoo.valid.txt',
    'test_data': 'datasets/yahoo_label_data/yahoo.test.txt',
    'label': True
}


params_ss_100={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'epochs': 1000,
    'test_nepoch': 5,
    'train_data': 'datasets/yahoo_label_data/yahoo.train.100.txt',
    'val_data': 'datasets/yahoo_label_data/yahoo.valid.txt',
    'test_data': 'datasets/yahoo_label_data/yahoo.test.txt',
    'vocab_file': 'datasets/yahoo_label_data/yahoo.vocab',
    'ncluster': 10,
    'label': True
}

params_ss_500={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'epochs': 1000,
    'test_nepoch': 5,
    'train_data': 'datasets/yahoo_label_data/yahoo.train.500.txt',
    'val_data': 'datasets/yahoo_label_data/yahoo.valid.txt',
    'test_data': 'datasets/yahoo_label_data/yahoo.test.txt',
    'vocab_file': 'datasets/yahoo_label_data/yahoo.vocab',
    'ncluster': 10,
    'label': True
}

params_ss_1000={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'epochs': 1000,
    'test_nepoch': 5,
    'train_data': 'datasets/yahoo_label_data/yahoo.train.1000.txt',
    'val_data': 'datasets/yahoo_label_data/yahoo.valid.txt',
    'test_data': 'datasets/yahoo_label_data/yahoo.test.txt',
    'vocab_file': 'datasets/yahoo_label_data/yahoo.vocab',
    'ncluster': 10,
    'label': True
}

params_ss_2000={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'epochs': 1000,
    'test_nepoch': 5,
    'train_data': 'datasets/yahoo_label_data/yahoo.train.2000.txt',
    'val_data': 'datasets/yahoo_label_data/yahoo.valid.txt',
    'test_data': 'datasets/yahoo_label_data/yahoo.test.txt',
    'vocab_file': 'datasets/yahoo_label_data/yahoo.vocab',
    'ncluster': 10,
    'label': True
}

params_ss_10000={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'epochs': 1000,
    'test_nepoch': 5,
    'train_data': 'datasets/yahoo_label_data/yahoo.train.10000.txt',
    'val_data': 'datasets/yahoo_label_data/yahoo.valid.txt',
    'test_data': 'datasets/yahoo_label_data/yahoo.test.txt',
    'vocab_file': 'datasets/yahoo_label_data/yahoo.vocab',
    'ncluster': 10,
    'label': True
}