params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 128,
    'enc_nh': 512,
    'dec_nh': 512,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'batch_size': 32,
    'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'datasets/short_yelp_data/short_yelp.train.txt',
    'val_data': 'datasets/short_yelp_data/short_yelp.valid.txt',
    'test_data': 'datasets/short_yelp_data/short_yelp.test.txt',
    "label": True
}


params_ss_100={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 128,
    'enc_nh': 512,
    'dec_nh': 512,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    # 'batch_size': 32,
    'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'datasets/short_yelp_data/short_yelp.train.100.txt',
    'val_data': 'datasets/short_yelp_data/short_yelp.valid.txt',
    'test_data': 'datasets/short_yelp_data/short_yelp.test.txt',
    'vocab_file': 'datasets/short_yelp_data/vocab.txt',
    'ncluster': 10,
    "label": True
}

params_ss_500={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 128,
    'enc_nh': 512,
    'dec_nh': 512,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    # 'batch_size': 32,
    'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'datasets/short_yelp_data/short_yelp.train.500.txt',
    'val_data': 'datasets/short_yelp_data/short_yelp.valid.txt',
    'test_data': 'datasets/short_yelp_data/short_yelp.test.txt',
    'vocab_file': 'datasets/short_yelp_data/vocab.txt',
    'ncluster': 10,
    "label": True
}

params_ss_1000={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 128,
    'enc_nh': 512,
    'dec_nh': 512,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    # 'batch_size': 32,
    'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'datasets/short_yelp_data/short_yelp.train.1000.txt',
    'val_data': 'datasets/short_yelp_data/short_yelp.valid.txt',
    'test_data': 'datasets/short_yelp_data/short_yelp.test.txt',
    'vocab_file': 'datasets/short_yelp_data/vocab.txt',
    'ncluster': 10,
    "label": True
}


params_ss_2000={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 128,
    'enc_nh': 512,
    'dec_nh': 512,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    # 'batch_size': 32,
    'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'datasets/short_yelp_data/short_yelp.train.2000.txt',
    'val_data': 'datasets/short_yelp_data/short_yelp.valid.txt',
    'test_data': 'datasets/short_yelp_data/short_yelp.test.txt',
    'vocab_file': 'datasets/short_yelp_data/vocab.txt',
    'ncluster': 10,
    "label": True
}


params_ss_10000={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 128,
    'enc_nh': 512,
    'dec_nh': 512,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    # 'batch_size': 32,
    'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'datasets/short_yelp_data/short_yelp.train.10000.txt',
    'val_data': 'datasets/short_yelp_data/short_yelp.valid.txt',
    'test_data': 'datasets/short_yelp_data/short_yelp.test.txt',
    'vocab_file': 'datasets/short_yelp_data/vocab.txt',
    'ncluster': 10,
    "label": True
}
