_base_ = ['configs/yolo_world/yolo_world_v2_t_4e_coco.py']

data_root = '/mnt/e/idfc_genai/owldino_data/'
metainfo = dict(classes=('signature', 'stamp'))

# Dataloaders: reuse the same JSON for train/val for now (small set)
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo))

test_dataloader = val_dataloader

# Training schedule tweaks for small data
train_cfg = dict(max_epochs=20)
optim_wrapper = dict(optimizer=dict(lr=1e-4))

# Two classes; set text categories for open-vocab head
model = dict(bbox_head=dict(num_classes=2, text_categories=('signature', 'stamp')))
