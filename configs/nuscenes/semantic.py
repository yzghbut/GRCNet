dataset_type = 'NuScenesDataset'

data_root = 'data/nuscenes'

num_classes = 16

ignore_index = -1

class_names = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
    'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
    'manmade', 'vegetation'
]

weight = None

resume = False

evaluate = True

test_only = False

seed = 55333843

save_path = 'outputs/nuscenes/semantic'

num_worker = 8

batch_size = 12

batch_size_val = 1

batch_size_test = 1

epoch = 50

eval_epoch = 50

clip_grad = 1.0

sync_bn = False

enable_amp = True

amp_dtype = 'float16'

empty_cache = False

empty_cache_per_epoch = False

find_unused_parameters = False

multiprocessing_sharing_strategy = 'file_system'

pin_memory = True

persistent_workers = True

prefetch_factor = 2

dataloader_timeout = 0

enable_wandb = False

wandb_project = 'LitePT'

wandb_key = None

mix_prob = 0.8

model = {
    'type':
    'DefaultSegmentorV2',
    'num_classes':
    num_classes,
    'backbone_out_channels':
    72,
    'backbone': {
        'type': 'LitePT',
        'in_channels': 4,
        'order': ('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        'stride': (2, 2, 2, 2),
        'enc_depths': (2, 2, 2, 6, 2),
        'enc_channels': (36, 72, 144, 252, 504),
        'enc_num_head': (2, 4, 8, 14, 28),
        'enc_patch_size': (1024, 1024, 1024, 1024, 1024),
        'enc_conv': (True, True, True, True, True),
        'enc_cpe_mode': ('full', 'full', 'full', 'bottleneck', 'bottleneck'),
        'enc_attn': (False, False, False, True, True),
        'dec_depths': (1, 1, 1, 1),
        'dec_channels': (72, 72, 144, 252),
        'dec_num_head': (4, 4, 8, 14),
        'dec_patch_size': (1024, 1024, 1024, 1024),
        'dec_conv': (True, True, True, True),
        'dec_cpe_mode': ('bottleneck', 'bottleneck', 'bottleneck', 'bottleneck'),
        'dec_attn': (False, False, False, False),
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
        'scene_query_num_classes': num_classes,
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
    'scene_query_semantic_weight':
    0.02,
    'scene_query_logit_fusion_init':
    0.08,
    'scene_query_logit_fusion_max':
    0.25,
    'scene_query_logit_fusion_adaptive':
    True,
    'scene_query_logit_fusion_adaptive_floor':
    0.25,
    'scene_query_logit_fusion_per_class':
    True,
    'scene_query_logit_fusion_class_scale_delta':
    0.5,
    'criteria': [{
        'type': 'CrossEntropyLoss',
        'loss_weight': 1.0,
        'ignore_index': ignore_index,
        'label_smoothing': 0.03
    }, {
        'type': 'LovaszLoss',
        'mode': 'multiclass',
        'loss_weight': 1.0,
        'ignore_index': ignore_index
    }]
}

optimizer = {'type': 'AdamW', 'lr': 0.002, 'weight_decay': 0.005}

scheduler = {
    'type':
    'OneCycleLR',
    'max_lr': [
        0.002, 0.0005, 0.0005, 0.00039999999999999996, 0.00039999999999999996,
        0.00019999999999999998, 0.00019999999999999998
    ],
    'pct_start':
    0.04,
    'anneal_strategy':
    'cos',
    'div_factor':
    10.0,
    'final_div_factor':
    100.0
}

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
    'weight_decay': 0.005
}]

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
        'transform': [{
            'type': 'RandomRotate',
            'angle': [-1, 1],
            'axis': 'z',
            'center': [0, 0, 0],
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
            'type': 'GridSample',
            'grid_size': 0.05,
            'hash_type': 'fnv',
            'mode': 'train',
            'return_grid_coord': True
        }, {
            'type': 'ToTensor'
        }, {
            'type': 'Update',
            'keys_dict': {
                'grid_size': 0.05
            }
        }, {
            'type': 'Collect',
            'keys': ('coord', 'grid_coord', 'segment', 'grid_size'),
            'feat_keys': ('coord', 'strength')
        }],
        'test_mode':
        False,
        'ignore_index':
        ignore_index,
        'raw_data_root':
        'data/nuscenes',
        'loop':
        1
    },
    'val': {
        'type':
        dataset_type,
        'split':
        'val',
        'data_root':
        data_root,
        'transform': [{
            'type': 'Copy',
            'keys_dict': {
                'segment': 'origin_segment'
            }
        }, {
            'type': 'GridSample',
            'grid_size': 0.05,
            'hash_type': 'fnv',
            'mode': 'train',
            'return_grid_coord': True,
            'return_inverse': True
        }, {
            'type': 'ToTensor'
        }, {
            'type': 'Collect',
            'keys': ('coord', 'grid_coord', 'segment', 'origin_segment', 'inverse'),
            'feat_keys': ('coord', 'strength')
        }],
        'test_mode':
        False,
        'ignore_index':
        ignore_index,
        'raw_data_root':
        'data/nuscenes'
    },
    'test': {
        'type':
        dataset_type,
        'split':
        'val',
        'data_root':
        data_root,
        'transform': [{
            'type': 'Copy',
            'keys_dict': {
                'segment': 'origin_segment'
            }
        }, {
            'type': 'GridSample',
            'grid_size': 0.025,
            'hash_type': 'fnv',
            'mode': 'train',
            'return_inverse': True
        }],
        'test_mode':
        True,
        'test_cfg': {
            'voxelize': {
                'type': 'GridSample',
                'grid_size': 0.05,
                'hash_type': 'fnv',
                'mode': 'test',
                'return_grid_coord': True
            },
            'crop':
            None,
            'post_transform': [{
                'type': 'ToTensor'
            }, {
                'type': 'Collect',
                'keys': ('coord', 'grid_coord', 'index'),
                'feat_keys': ('coord', 'strength')
            }],
            'aug_transform': [[{
                'type': 'RandomScale',
                'scale': [0.9, 0.9]
            }], [{
                'type': 'RandomScale',
                'scale': [0.95, 0.95]
            }], [{
                'type': 'RandomScale',
                'scale': [1.0, 1.0]
            }], [{
                'type': 'RandomScale',
                'scale': [1.05, 1.05]
            }], [{
                'type': 'RandomScale',
                'scale': [1.1, 1.1]
            }], [{
                'type': 'RandomScale',
                'scale': [0.9, 0.9]
            }, {
                'type': 'RandomFlip',
                'p': 1
            }], [{
                'type': 'RandomScale',
                'scale': [0.95, 0.95]
            }, {
                'type': 'RandomFlip',
                'p': 1
            }], [{
                'type': 'RandomScale',
                'scale': [1.0, 1.0]
            }, {
                'type': 'RandomFlip',
                'p': 1
            }], [{
                'type': 'RandomScale',
                'scale': [1.05, 1.05]
            }, {
                'type': 'RandomFlip',
                'p': 1
            }], [{
                'type': 'RandomScale',
                'scale': [1.1, 1.1]
            }, {
                'type': 'RandomFlip',
                'p': 1
            }]]
        },
        'ignore_index':
        ignore_index,
        'raw_data_root':
        'data/nuscenes'
    }
}

hooks = [{
    'type': 'CheckpointLoader'
}, {
    'type': 'ModelEMAHook',
    'decay': 0.9997,
    'warmup': True
}, {
    'type': 'ModelHook'
}, {
    'type': 'IterationTimer',
    'warmup_iter': 2
}, {
    'type': 'InformationWriter'
}, {
    'type': 'SemSegEvaluator'
}, {
    'type': 'SemSegEvaluator',
    'use_ema': True
}, {
    'type': 'CheckpointSaver',
    'save_freq': 5
}, {
    'type': 'PreciseEvaluator',
    'test_last': False,
    'evaluate_both': True
}]

train = {'type': 'DefaultTrainer'}

test = {'type': 'SemSegTester', 'verbose': True}

ema_distill = {
    'enabled': True,
    'interval': 4,
    'confidence_threshold': 0.8,
    'reliability_floor': 0.25,
    'ignore_index': ignore_index,
    'max_weight': 0.05,
    'final_weight': 0.02,
    'warmup_ratio': 0.1,
    'ramp_end_ratio': 0.2,
    'decay_start_ratio': 0.8
}
