### Three #'s for my comments; single # comments are the author's
import all_constants as ac


def base():
    config = {}

    config['embed_dim'] = 512
    config['ff_dim'] = 2048
    config['num_enc_layers'] = 6
    config['num_dec_layers'] = 6
    config['num_heads'] = 8

    # architecture
    config['use_bias'] = True
    config['fix_norm'] = True
    config['scnorm'] = True
    config['mask_logit'] = True
    config['pre_act'] = True
    config['mha_sup'] = False

    config['clip_grad'] = 1.0
    config['lr_scheduler'] = ac.NO_WU
    config['warmup_steps'] = 8000
    config['lr'] = 3e-4
    config['lr_scale'] = 1.
    config['lr_decay'] = 0.8
    config['stop_lr'] = 5e-5
    config['patience'] = 3
    config['alpha'] = 0.7
    config['label_smoothing'] = 0.1
    config['batch_size'] = 4096
    config['epoch_size'] = 1000
    config['max_epochs'] = 200
    config['dropout'] = 0.3
    config['att_dropout'] = 0.3
    config['ff_dropout'] = 0.3
    config['word_dropout'] = 0.1
    config['seq_len_threshold'] = None

    # Decoding
    config['beam_size'] = 4
    config['beam_alpha'] = 0.6

    return config


def en2vi():
    config = base()
    config['epoch_size'] = 1500
    config['scnorm'] = False
    config['warmup_steps'] = 8000
    config['lr_scheduler'] = ac.UPFLAT_WU
    config['stop_lr'] = 1e-5
    config['mha_sup'] = True
    config['seq_len_threshold'] = 72
    
    return config


def ar2en():
    config = base()
    config['epoch_size'] = 2000
    config['scnorm'] = False
    config['warmup_steps'] = 8000
    config['lr_scheduler'] = ac.UPFLAT_WU
    config['stop_lr'] = 1e-5
    config['mha_sup'] = True
    config['seq_len_threshold'] = 75
    
    return config


def en2he():
    config = base()
    config['epoch_size'] = 2000
    config['scnorm'] = False
    config['warmup_steps'] = 8000
    config['lr_scheduler'] = ac.UPFLAT_WU
    config['stop_lr'] = 1e-5
    config['mha_sup'] = True
    config['seq_len_threshold'] = 72
    
    return config


def gl2en():
    config = base()
    config['epoch_size'] = 100
    config['scnorm'] = False
    config['warmup_steps'] = 8000
    config['lr_scheduler'] = ac.UPFLAT_WU
    config['mha_sup'] = True
    config['seq_len_threshold'] = 79
    config['max_epochs'] = 1000
    config['dropout'] = 0.4
    config['att_dropout'] = 0.4
    config['ff_dropout'] = 0.4
    config['num_heads'] = 4
    config['num_enc_layers'] = 4
    config['num_dec_layers'] = 4
    
    return config


def sk2en():
    config = base()
    config['epoch_size'] = 600
    config['scnorm'] = False
    config['warmup_steps'] = 8000
    config['lr_scheduler'] = ac.UPFLAT_WU
    config['stop_lr'] = 1e-5
    config['mha_sup'] = True
    
    config['seq_len_threshold'] = 75
    
    return config



