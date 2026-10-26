# InstanceParser runs after geometric transforms and cropping.
dataset_type = 'ScanNetDataset'

data_root = 'data/scannet'

num_classes = 20

ignore_index = -1

class_names = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink',
    'bathtub', 'otherfurniture'
]

segment_ignore_index = (-1, 0, 1)

weight = None

resume = False

evaluate = True

test_only = False

seed = 55333841

save_path = 'outputs/scannet/instance'

num_worker = 8

batch_size = 4

batch_size_val = 1

batch_size_test = 1

epoch = 800

eval_epoch = 100

clip_grad = 1.0

sync_bn = False

enable_amp = True

amp_dtype = 'float16'

empty_cache = False

empty_cache_per_epoch = False

find_unused_parameters = False

multiprocessing_sharing_strategy = 'file_system'

pin_memory = True

persistent_workers = False

prefetch_factor = 1

dataloader_timeout = 0

enable_wandb = False

wandb_project = 'LitePT'

wandb_key = None

mix_prob = 0.8

optimizer = {'type': 'AdamW', 'lr': 0.002, 'weight_decay': 0.05}

param_dicts = [{
    'keyword': 'scene_query',
    'lr': 0.0005,
    'weight_decay': 0.01
}, {
    'keyword': 'reliability_gate_logit',
    'lr': 0.0005,
    'weight_decay': 0.0
}, {
    'keywords': ('raw_distance_gamma', 'trust_gate', 'raw_uncertainty_trust'),
    'lr': 0.00039999999999999996,
    'weight_decay': 0.0
}, {
    'keywords': ('key_spectral_gate', 'descriptor_key'),
    'lr': 0.00039999999999999996,
    'weight_decay': 0.01
}, {
    'keyword': 'geobias',
    'lr': 0.00019999999999999998,
    'weight_decay': 0.01
}, {
    'keyword': 'block',
    'lr': 0.00019999999999999998,
    'weight_decay': 0.05
}]

scheduler = {
    'type':
    'OneCycleLR',
    'max_lr': [
        0.002, 0.0005, 0.0005, 0.00039999999999999996, 0.00039999999999999996,
        0.00019999999999999998, 0.00019999999999999998
    ],
    'pct_start':
    0.05,
    'anneal_strategy':
    'cos',
    'div_factor':
    10.0,
    'final_div_factor':
    1000.0
}

train = {'type': 'DefaultTrainer'}

model = {
    'type':
    'PG-v1m2',
    'backbone': {
        'type': 'LitePT',
        'in_channels': 6,
        'order': ('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        'stride': (2, 2, 2, 2),
        'enc_depths': (2, 2, 2, 6, 2),
        'enc_channels': (36, 72, 144, 252, 504),
        'enc_num_head': (2, 4, 8, 14, 28),
        'enc_patch_size': (1024, 1024, 1024, 1024, 1024),
        'enc_conv': (True, True, True, False, False),
        'enc_cpe_mode': ('full', 'full', 'full', 'full', 'full'),
        'enc_attn': (False, False, False, True, True),
        'dec_depths': (2, 2, 2, 2),
        'dec_channels': (72, 72, 144, 252),
        'dec_num_head': (4, 4, 8, 14),
        'dec_patch_size': (1024, 1024, 1024, 1024),
        'dec_conv': (True, True, True, False),
        'dec_cpe_mode': ('full', 'full', 'full', 'full'),
        'dec_attn': (False, False, False, True),
        'dec_reliability_gate': True,
        'dec_reliability_floor': 0.5,
        'dec_reliability_gate_init': -3.0,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'qk_scale': None,
        'attn_drop': 0.0,
        'proj_drop': 0.0,
        'drop_path': 0.3,
        'shuffle_orders': True,
        'pre_norm': True,
        'enc_mode': False,
        'cpe_ratio': 0.125,
        'cpe_channel_multiple': 8,
        'geobias_k': (4, 6, 10),
        'geobias_num_frequencies': 2,
        'geobias_descriptor_rank': 2,
        'geobias_head_dim_multiple': 8,
        'geobias_bias_scale': None,
        'geobias_trust_region': 0.1,
        'geobias_distance_rank': 2,
        'geobias_distance_gamma_max': 0.05,
        'geobias_distance_stage_strengths': (0.0, 0.0, 0.0, 0.0, 1.0),
        'geobias_distance_gamma_init': -2.0,
        'geobias_distance_projection_init': 0.05,
        'geobias_uncertainty_budget': 0.1,
        'geobias_uncertainty_init': -3.0,
        'enable_alibi': True,
        'alibi_local_fraction': 0.5,
        'scene_query_num_tokens': 16,
        'scene_query_rank': 32,
        'scene_query_coord_frequencies': 4,
        'scene_query_num_classes': 0,
        'scene_query_assignment_mode': 'unbalanced_sinkhorn',
        'scene_query_sinkhorn_iterations': 12,
        'scene_query_sinkhorn_epsilon': 0.5,
        'scene_query_sinkhorn_mass_relaxation': 0.5,
        'scene_query_layer_scale_init': 0.05,
        'scene_query_use_geometry_reliability': True,
        'scene_query_geometry_k': (4, 6, 10),
        'scene_query_shared_basis': True,
        'scene_query_replace_ffn': True,
        'scene_query_ffn_rank': 256,
        'scene_query_output_adapter_rank': 32,
        'scene_query_local_reliability_floor': 0.5,
        'scene_query_global_reliability_floor': 0.5
    },
    'backbone_out_channels':
    72,
    'semantic_num_classes':
    num_classes,
    'semantic_ignore_index':
    ignore_index,
    'segment_ignore_index':
    segment_ignore_index,
    'instance_ignore_index':
    ignore_index,
    'cluster_thresh':
    1.5,
    'cluster_closed_points':
    600,
    'cluster_propose_points':
    200,
    'cluster_min_points':
    50,
    'voxel_size':
    0.02,
    'criteria': [{
        'type': 'CrossEntropyLoss',
        'loss_weight': 1.0,
        'ignore_index': ignore_index
    }, {
        'type': 'LovaszLoss',
        'mode': 'multiclass',
        'loss_weight': 1.0,
        'ignore_index': ignore_index
    }]
}

data = {
    'num_classes': num_classes,
    'ignore_index': ignore_index,
    'names': class_names,
    'train': {
        'type':
        dataset_type,
        'split':
        'train',
        'data_root':
        data_root,
        'strict_data':
        True,
        'test_mode':
        False,
        'transform': [{
            'type': 'CenterShift',
            'apply_z': True
        }, {
            'type': 'RandomDropout',
            'dropout_ratio': 0.2,
            'dropout_application_ratio': 0.5
        }, {
            'type': 'RandomRotate',
            'angle': [-1, 1],
            'axis': 'z',
            'center': [0, 0, 0],
            'p': 0.5
        }, {
            'type': 'RandomRotate',
            'angle': [-0.015625, 0.015625],
            'axis': 'x',
            'p': 0.5
        }, {
            'type': 'RandomRotate',
            'angle': [-0.015625, 0.015625],
            'axis': 'y',
            'p': 0.5
        }, {
            'type': 'RandomScale',
            'scale': [0.9, 1.1]
        }, {
            'type': 'RandomFlip',
            'p': 0.5
        }, {
            'type': 'RandomJitter',
            'sigma': 0.005,
            'clip': 0.02
        }, {
            'type': 'ElasticDistortion',
            'distortion_params': [[0.2, 0.4], [0.8, 1.6]]
        }, {
            'type': 'ChromaticAutoContrast',
            'p': 0.2,
            'blend_factor': None
        }, {
            'type': 'ChromaticTranslation',
            'p': 0.95,
            'ratio': 0.1
        }, {
            'type': 'ChromaticJitter',
            'p': 0.95,
            'std': 0.05
        }, {
            'type': 'GridSample',
            'grid_size': 0.02,
            'hash_type': 'fnv',
            'mode': 'train',
            'return_grid_coord': True
        }, {
            'type': 'SphereCrop',
            'point_max': 102400,
            'mode': 'random'
        }, {
            'type': 'CenterShift',
            'apply_z': False
        }, {
            'type': 'NormalizeColor'
        }, {
            'type': 'InstanceParser',
            'segment_ignore_index': segment_ignore_index,
            'instance_ignore_index': ignore_index
        }, {
            'type': 'ToTensor'
        }, {
            'type': 'Update',
            'keys_dict': {
                'grid_size': 0.02
            }
        }, {
            'type':
            'Collect',
            'keys': ('coord', 'grid_coord', 'segment', 'instance', 'instance_centroid', 'bbox',
                     'grid_size'),
            'feat_keys': ('color', 'normal')
        }]
    },
    'val': {
        'type':
        dataset_type,
        'split':
        'val',
        'data_root':
        data_root,
        'strict_data':
        True,
        'test_mode':
        False,
        'transform': [{
            'type': 'CenterShift',
            'apply_z': True
        }, {
            'type': 'Copy',
            'keys_dict': {
                'coord': 'origin_coord',
                'segment': 'origin_segment',
                'instance': 'origin_instance'
            }
        }, {
            'type': 'GridSample',
            'grid_size': 0.02,
            'hash_type': 'fnv',
            'mode': 'train',
            'return_grid_coord': True
        }, {
            'type': 'CenterShift',
            'apply_z': False
        }, {
            'type': 'NormalizeColor'
        }, {
            'type': 'InstanceParser',
            'segment_ignore_index': segment_ignore_index,
            'instance_ignore_index': ignore_index
        }, {
            'type': 'ToTensor'
        }, {
            'type':
            'Collect',
            'keys': ('coord', 'grid_coord', 'segment', 'instance', 'origin_coord', 'origin_segment',
                     'origin_instance', 'instance_centroid', 'bbox'),
            'feat_keys': ('color', 'normal'),
            'offset_keys_dict': {
                'offset': 'coord',
                'origin_offset': 'origin_coord'
            }
        }]
    },
    'test': {}
}

hooks = [{
    'type': 'CheckpointLoader'
}, {
    'type': 'ModelHook'
}, {
    'type': 'IterationTimer',
    'warmup_iter': 2
}, {
    'type': 'InformationWriter'
}, {
    'type': 'InsSegEvaluator',
    'segment_ignore_index': segment_ignore_index,
    'instance_ignore_index': ignore_index
}, {
    'type': 'CheckpointSaver',
    'save_freq': 10
}]

ema_distill = {'enabled': False}

test = {}
