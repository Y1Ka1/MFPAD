_base_ = ["./MomAD_small_stage2_roboAD_6s.py"]

fut_ts = 24
ego_fut_ts = 24

model = dict(
    head=dict(
        motion_plan_head=dict(
            fut_ts=fut_ts,
            ego_fut_ts=ego_fut_ts,
            motion_anchor="data/kmeans/kmeans_motion_6_12s.npy",
            plan_anchor="data/kmeans/kmeans_plan_6_12s.npy",
            refine_layer=dict(
                fut_ts=fut_ts,
                ego_fut_ts=ego_fut_ts,
            ),
            planning_sampler=dict(
                ego_fut_ts=ego_fut_ts,
            ),
            planning_decoder=dict(
                ego_fut_ts=ego_fut_ts,
            ),
        )
    )
)

dataset_type = "NuScenes3DDataset_roboAD_12s"
anno_root = "data/infos/" if version == "trainval" else "data/infos/mini/"

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    map_classes=map_class_names,
    modality=input_modality,
    version="v1.0-trainval",
)

eval_config = dict(
    **data_basic_config,
    ann_file=anno_root + "nuscenes_infos_val_12s.pkl",
    pipeline=eval_pipeline,
    test_mode=True,
    future_horizon_sec=12.0,
    step_sec=0.5,
    eval_output_dir="work_dirs/extended_horizon_12s",
    eval_output_prefix="mfpad_12s",
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_train_12s.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val_12s.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val_12s.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
    ),
)
