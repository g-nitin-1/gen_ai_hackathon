# Copyright 2025 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small OWLv2 finetune config for signature/stamp COCO TFRecords."""

import ml_collections


def _train_preproc(input_size: int, num_instances: int, max_queries: int):
  """Lightweight training pipeline."""
  return (
      f'keep(boxes, crowd, image, instance_labels, instance_text_labels, '
      f'negative_labels, negative_text_labels, _seed, seed)'
      f'|random_flip_left_right'
      f'|resize_with_pad(size={input_size})'
      f'|canonicalize_text_labels'
      f'|crop_or_pad({input_size}, {num_instances})'
      f'|crop_or_pad_meta_data({num_instances}, {num_instances})'
      f'|single_to_multi_label(max_num_labels={num_instances})'
      f'|add_query_set(lower=True, max_queries={max_queries}, '
      f'include_negatives=True)'
      f'|clip_tokenize_queries(max_token_len=16)')


def _eval_preproc(input_size: int, num_instances: int, max_queries: int):
  """Eval pipeline mirrors train but without augmentation."""
  return (
      f'resize_with_pad(size={input_size})'
      f'|canonicalize_text_labels'
      f'|crop_or_pad({input_size}, {num_instances})'
      f'|crop_or_pad_meta_data({num_instances}, {num_instances})'
      f'|single_to_multi_label(max_num_labels={num_instances})'
      f'|add_query_set(lower=True, max_queries={max_queries}, '
      f'include_negatives=True)'
      f'|clip_tokenize_queries(max_token_len=16)')


def get_config():
  """Returns config for finetuning OWLv2 on the sig/stamp TFRecords."""
  cfg = ml_collections.ConfigDict()
  cfg.experiment_name = 'owl_vit_sigstamp'

  # Dataset.
  cfg.dataset_name = 'owl_vit'
  cfg.dataset_configs = ml_collections.ConfigDict()
  cfg.dataset_configs.input_size = 640
  cfg.dataset_configs.num_instances = 50
  cfg.dataset_configs.max_queries = 50
  cfg.dataset_configs.max_query_length = 16
  cfg.dataset_configs.prefetch_to_device = 1

  # File patterns to override via flags.
  cfg.dataset_configs.train_file_pattern = ''
  cfg.dataset_configs.validation_file_pattern = ''
  cfg.dataset_configs.class_names = ['signature', 'stamp']
  cfg.dataset_configs.decoder_kwarg_list = ({'name': 'coco_sigstamp:0.0.1'},)

  cfg.dataset_configs.train = ml_collections.ConfigDict()
  cfg.dataset_configs.train.preproc_spec = _train_preproc(
      cfg.dataset_configs.input_size,
      cfg.dataset_configs.num_instances,
      cfg.dataset_configs.max_queries)
  cfg.dataset_configs.train.mosaic_sizes = (1,)
  cfg.dataset_configs.train.mosaic_probs = (1.0,)

  cfg.dataset_configs.eval = ml_collections.ConfigDict()
  cfg.dataset_configs.eval.preproc_spec = _eval_preproc(
      cfg.dataset_configs.input_size,
      cfg.dataset_configs.num_instances,
      cfg.dataset_configs.max_queries)

  # Model + matcher.
  cfg.matcher = 'hungarian_cover_tpu'
  cfg.model = ml_collections.ConfigDict()
  cfg.model.normalize = True
  cfg.model.body = ml_collections.ConfigDict()
  cfg.model.body.type = 'clip'
  cfg.model.body.variant = 'vit_b32'
  cfg.model.body.merge_class_token = 'mul-ln'
  cfg.model.box_bias = 'both'

  # Training.
  cfg.batch_size = 1
  cfg.eval_batch_size = 1
  cfg.num_training_steps = 2000
  cfg.rng_seed = 0
  cfg.log_loss_every_steps = 50
  cfg.eval_every_steps = 200
  cfg.checkpoint_every_steps = 200
  cfg.gradient_clip_norm = 1.0
  cfg.data_dtype_str = 'float32'
  cfg.optimizer = 'adamw'
  cfg.base_lr = 1e-4
  cfg.weight_decay = 1e-4

  # Init from HF snapshot (override via flag).
  cfg.init_from = ml_collections.ConfigDict()
  cfg.init_from.checkpoint_path = ''

  return cfg
