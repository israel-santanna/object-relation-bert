# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re
import random

import modeling
import tokenization
import tensorflow as tf

import math
import numpy as np
from scipy.spatial.distance import cosine as cos_distance
from scipy.stats import pearsonr

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")

flags.DEFINE_float("fake_context_prob", 0.5, "Percentage of documents that should have one of the objects changed to another one not in the image, creating a fake context.")

flags.DEFINE_bool("do_predict", False, "Whether to run prediction.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")


class InputExample(object):

  def __init__(self, unique_id, text, bounding_boxes):
    self.unique_id = unique_id
    self.text = text
    self.bounding_boxes = bounding_boxes


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, bounding_boxes, fake_context_label, input_mask, input_type_ids, masked_id, masked_pos, tokens_size):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.bounding_boxes=bounding_boxes
    self.fake_context_label=fake_context_label
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids
    self.masked_id=masked_id
    self.masked_pos=masked_pos
    self.tokens_size=tokens_size


def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_bounding_boxes = []
  all_fake_context_labels = []
  all_input_mask = []
  all_input_type_ids = []
  all_masked_ids=[]
  all_masked_positions=[]

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_bounding_boxes.append(feature.bounding_boxes)
    all_fake_context_labels.append([feature.fake_context_label])
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)
    all_masked_ids.append([feature.masked_id])
    all_masked_positions.append([feature.masked_pos])

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "bounding_boxes":
            tf.constant(
                all_bounding_boxes, shape=[num_examples, seq_length, 4],
                dtype=tf.float32),
        "fake_context_labels":
            tf.constant(
                all_fake_context_labels, shape=[num_examples, 1],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "masked_ids":
            tf.constant(
                all_masked_ids,
                shape=[num_examples, 1],
                dtype=tf.int32),
        "masked_positions":
            tf.constant(
                all_masked_positions,
                shape=[num_examples, 1],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=True)
    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    bounding_boxes = features["bounding_boxes"]
    fake_context_labels = features["fake_context_labels"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]
    masked_ids = features["masked_ids"]
    masked_positions = features["masked_positions"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        bounding_boxes=bounding_boxes,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    if use_tpu:
      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)


    predictions = {
        "unique_id": unique_ids,
    }

    all_layers = model.get_all_encoder_layers()
    for (i, layer_index) in enumerate(layer_indexes):
      predictions["layer_output_%d" % i] = all_layers[layer_index]
    
    (cluster_loss, cluster_example_loss,
      cluster_fc_predictions) = get_cluster_distance_output(all_layers,
         layer_indexes, fake_context_labels)
    predictions["cluster_distance"] = cluster_example_loss

    (masked_lm_loss, masked_lm_example_loss,
      masked_lm_log_probs, masked_lm_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_positions, masked_ids)
    predictions["mlm_probs"] = masked_lm_probs

    output_layer = model.get_pooled_output()
    (fake_context_loss, fake_context_example_loss,
     fake_context_log_probs, fake_context_probs) = get_fake_context_output(
         bert_config, model.get_pooled_output(), fake_context_labels)
    predictions["fake_context_probs"] = fake_context_probs

    total_loss = masked_lm_loss + fake_context_loss + cluster_loss

    if mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    fake_context_example_loss, fake_context_log_probs,
                    fake_context_labels, cluster_example_loss,
                    cluster_fc_predictions):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss)

        fake_context_log_probs = tf.reshape(
            fake_context_log_probs, [-1, fake_context_log_probs.shape[-1]])
        fake_context_predictions = tf.argmax(
            fake_context_log_probs, axis=-1, output_type=tf.int32)
        fake_context_labels = tf.reshape(fake_context_labels, [-1])
        fake_context_accuracy = tf.metrics.accuracy(
            labels=fake_context_labels, predictions=fake_context_predictions)
        fake_context_mean_loss = tf.metrics.mean(
            values=fake_context_example_loss)

        cluster_accuracy = tf.metrics.accuracy(
            labels=fake_context_labels, predictions=cluster_fc_predictions)
        cluster_mean_loss = tf.metrics.mean(
            values=cluster_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "fake_context_accuracy": fake_context_accuracy,
            "fake_context_loss": fake_context_mean_loss,
            "cluster_accuracy": cluster_accuracy,
            "cluster_loss": cluster_mean_loss
        }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_ids,
          fake_context_example_loss, fake_context_log_probs,
          fake_context_labels, cluster_example_loss, cluster_fc_predictions
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)

    else:
      raise ValueError("Only EVAL and PREDICT modes are supported: %s"%(mode))

    return output_spec

  return model_fn

def get_cluster_distance_output(all_layers, layer_indexes, fake_context_labels, threshold=0.5):
  layers = []
  for (i, layer_index) in enumerate(layer_indexes):
    layers.append(all_layers[layer_index])

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   S = tensor sequence length
  #   E = embedding size = number of layers * hidden size
  # [B, S, E]
  embeddings = tf.concat(layers, axis=2)
  embeddings_shape = modeling.get_shape_list(embeddings)
  batch_size = embeddings_shape[0]
  seq_length = embeddings_shape[1]
  embedding_size = embeddings_shape[2]

  embeddings = tf.nn.l2_normalize(embeddings, axis=2)
  tokens = tf.split(embeddings, num_or_size_splits=seq_length, axis=1)
  per_example_loss = tf.zeros([batch_size, 1], tf.float32)
  for i, token in enumerate(tokens):
    distance = 1 - tf.matmul(token, embeddings, transpose_b=True)
    token_loss = tf.reduce_mean(distance, axis=-1)
    per_example_loss += token_loss
  per_example_loss /= seq_length
  per_example_loss = tf.check_numerics(per_example_loss, "per_example_loss division error")

  per_example_loss = tf.where(tf.equal(fake_context_labels, 1),
                              -per_example_loss, per_example_loss)

  predictions = tf.where(tf.greater(per_example_loss, threshold),
                         tf.ones(per_example_loss.shape, dtype=tf.int32),
                         tf.zeros(per_example_loss.shape, dtype=tf.int32))

  loss = tf.reduce_mean(per_example_loss)
  
  return (loss, per_example_loss, predictions)

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids):
  output_layer = gather_indexes(input_tensor, positions)
  with tf.variable_scope("cls/mlm_predictions"):
    output_layer = tf.layers.dense(
        output_layer,
        units=bert_config.hidden_size,
        activation=modeling.get_activation(bert_config.hidden_act),
        kernel_initializer=modeling.create_initializer(
            bert_config.initializer_range))
    output_layer = modeling.layer_norm(output_layer)

    output_bias = tf.get_variable(
      "output_bias", [bert_config.vocab_size], initializer=tf.zeros_initializer())
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    probs = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])

    loss = tf.reduce_mean(per_example_loss)

  return (loss, per_example_loss, log_probs, probs)

def get_fake_context_output(bert_config, input_tensor, labels):
  with tf.variable_scope("cls/ctx_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    probs = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
  return (loss, per_example_loss, log_probs, probs)

def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor

def get_bounding_boxes(line):
  bboxes = [map(float, i.strip().split(',')) for i in line.split(' ')]
  return bboxes

def convert_examples_to_features(examples, seq_length, tokenizer,
  fake_context_prob, rng):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  unique_id = -1
  # Get a out_context_prob sample of the document sentences to change a random
  # object for an out of context object
  n_fake_context = int(round(fake_context_prob * len(examples)))
  fake_context_idxs = rng.sample(xrange(len(examples)), n_fake_context)
  # Sort the indexes to avoid having to look through the whole list each time
  fake_context_idxs.sort()
  has_fake_context_idx = 0
  vocab_words = list(tokenizer.vocab.keys())

  for (ex_index, example) in enumerate(examples):
    ex_tokens = tokenizer.tokenize(example.text)

    fake_context_label = 0
    if (has_fake_context_idx < n_fake_context
      and ex_index == fake_context_idxs[has_fake_context_idx]):
      fake_context_label = 1
      has_fake_context_idx += 1
    for i in xrange(len(ex_tokens)):
      unique_id += 1
      masked_id = tokenizer.convert_tokens_to_ids([ex_tokens[i]])[0]
      masked_pos = i + 1

      tokens = ["[CLS]"]
      tokens.extend(ex_tokens)
      tokens[masked_pos] = "[MASK]"
      tokens.append("[SEP]")
      bounding_boxes = [[0.0,0.0,0.0,0.0]]
      bounding_boxes.extend(get_bounding_boxes(example.bounding_boxes))
      bounding_boxes.append([0.0,0.0,0.0,0.0])
      input_type_ids = [0]*len(tokens)

      if fake_context_label:
        while True:
          new_token_idx = rng.randint(1, len(tokens) - 2)
          if new_token_idx != i + 1:
            break
        while True:
          # The first 6 tokens in the vocab are special tokens ([CLS],[SEP],etc.)
          new_token = vocab_words[rng.randint(6, len(vocab_words) - 1)]
          # If the sentence length is bigger than the vocabulary,
          # there might be one of each token in the sentence and
          # the loop will never break.
          # If that is the case, we just check for a token different from the
          # original, otherwise, we look for a token not in the sentence yet.
          if seq_length - 2 >= (len(vocab_words) - 6):
            if tokens[new_token_idx] != new_token:
              break
          elif new_token not in tokens:
            break
        tokens[new_token_idx] = new_token

      # The convention in BERT is:
      # (a) For sequence pairs:
      #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
      #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
      # (b) For single sequences:
      #  tokens:   [CLS] the dog is hairy . [SEP]
      #  type_ids: 0     0   0   0  0     0 0
      #
      # Where "type_ids" are used to indicate whether this is the first
      # sequence or the second sequence. The embedding vectors for `type=0` and
      # `type=1` were learned during pre-training and are added to the wordpiece
      # embedding vector (and position vector). This is not *strictly* necessary
      # since the [SEP] token unambiguously separates the sequences, but it makes
      # it easier for the model to learn the concept of sequences.
      #
      # For classification tasks, the first vector (corresponding to [CLS]) is
      # used as as the "sentence vector". Note that this only makes sense because
      # the entire model is fine-tuned.

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)
        bounding_boxes.append([0.0,0.0,0.0,0.0])

      assert len(input_ids) == seq_length
      assert len(input_mask) == seq_length
      assert len(input_type_ids) == seq_length
      assert len(bounding_boxes) == seq_length

      features.append(
          InputFeatures(
              unique_id=unique_id,
              tokens=tokens,
              input_ids=input_ids,
              bounding_boxes=bounding_boxes,
              fake_context_label=fake_context_label,
              input_mask=input_mask,
              input_type_ids=input_type_ids,
              masked_id=masked_id,
              masked_pos=masked_pos,
              tokens_size=len(ex_tokens)+2))
  return features

def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  with tf.gfile.GFile(input_file, "r") as reader:
    first_line = True
    while True:
      line = tokenization.convert_to_unicode(reader.readline())
      if not line:
        break
      line = line.strip()

      if not line:
        first_line = True
      else:
        if first_line:
          first_line = False
        else:
          tokens_line, bbox_line = line.split('|')
          tokens_line = tokens_line.strip()
          bbox_line = bbox_line.strip()
          examples.append(
              InputExample(unique_id=unique_id, text=tokens_line, bounding_boxes=bbox_line))
          unique_id += 1
  return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  examples = read_examples(FLAGS.input_file)

  rng = random.Random(FLAGS.random_seed)
  features = convert_examples_to_features(
      examples=examples, seq_length=FLAGS.max_seq_length,
      tokenizer=tokenizer, fake_context_prob=FLAGS.fake_context_prob, rng=rng)

  unique_id_to_feature = {}
  for feature in features:
    unique_id_to_feature[feature.unique_id] = feature

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      layer_indexes=layer_indexes,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size)

  input_fn = input_fn_builder(
      features=features, seq_length=FLAGS.max_seq_length)

  if FLAGS.do_eval:
    print("***** Running evaluation *****")
    print("Batch size = " + str(FLAGS.batch_size))

    result = estimator.evaluate(
        input_fn=input_fn, steps=FLAGS.max_eval_steps)

    # output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    # with tf.gfile.GFile(output_eval_file, "w") as writer:
    print("***** Eval results *****")
    for key in sorted(result.keys()):
      print(key + " = " + str(result[key]))
        # writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    # with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file,
    #                                              "w")) as writer:
    print("***** Running prediction *****")
    print("Number of examples: " + str(len(examples)))
    print("Number of masked examples (features): " +
      str(len(features)))
    accuracy_vector = [0] * bert_config.vocab_size
    mlm_conf_matrix = np.zeros((bert_config.vocab_size,bert_config.vocab_size), dtype=np.int32)
    fc_conf_matrix = np.zeros((2,2), dtype=np.int32)
    mlm_length_accuracy = np.zeros((12,2), dtype=np.int32)
    fc_length_accuracy = np.zeros((12,2), dtype=np.int32)
    mlm_prob_vector = [0] * 100
    fc_prob_vector = [0] * 100
    dist_prob_vector = [[0] * 100, [0] * 100]
    mlm_accuracy = 0.0
    fc_accuracy = 0.0
    len_results = 0.0
    total_cos_mean_dist = [0.0, 0.0]
    total_p_mean_dist = [0.0, 0.0]
    for result in estimator.predict(input_fn, yield_single_examples=True):
      unique_id = int(result["unique_id"])
      feature = unique_id_to_feature[unique_id]
      mlm_sorted_inds = np.argsort(-result["mlm_probs"])
      ground_truth_pos = np.where(mlm_sorted_inds == feature.masked_id)[0][0]
      accuracy_vector[ground_truth_pos] += 1
      prob_pos = int(math.floor(result["mlm_probs"][feature.masked_id] / 0.01))
      if prob_pos >= 100:
        prob_pos = 99
      mlm_prob_vector[prob_pos] += 1
      prob_pos = int(math.floor(result["fake_context_probs"][feature.fake_context_label] / 0.01))
      if prob_pos >= 100:
        prob_pos = 99
      fc_prob_vector[prob_pos] += 1
      prob_pos = int(math.floor(abs(result["cluster_distance"][0]) / 0.01))
      if prob_pos >= 100:
        prob_pos = 99
      dist_prob_vector[feature.fake_context_label][prob_pos] += 1
      i = 0
      token_id = -1
      while token_id < 6:
        token_id = mlm_sorted_inds[i]
        i += 1
      mlm_is_accurate = int(token_id == feature.masked_id)
      mlm_accuracy += mlm_is_accurate
      mlm_length_accuracy[feature.tokens_size-1, mlm_is_accurate] += 1 
      mlm_conf_matrix[feature.masked_id, mlm_sorted_inds[0]] += 1

      fc_sorted_inds = np.argsort(-result["fake_context_probs"])
      fc_is_accurate = int(fc_sorted_inds[0] == feature.fake_context_label)
      fc_accuracy += fc_is_accurate
      fc_length_accuracy[feature.tokens_size-1, fc_is_accurate] += 1 
      fc_conf_matrix[feature.fake_context_label, fc_sorted_inds[0]] += 1

      embeddings = []
      for i in xrange(1, feature.tokens_size):
        all_layers = []
        for (j, layer_index) in enumerate(layer_indexes):
          layer_output = result["layer_output_%d" % j]
          all_layers.append([float(x) for x in layer_output[i:(i + 1)].flat])
        embeddings.append(np.array(all_layers).flatten())
      # Embedding Distance:
      img_cos_mean_dist = [0.0, 0.0]
      img_p_mean_dist = [0.0, 0.0]
      for (i, embedding_i) in enumerate(embeddings):
        cos_mean_dist = 0.0
        p_mean_dist = 0.0
        for (j, embedding_j) in enumerate(embeddings):
          if j == 0:
            continue
          cos_dist = cos_distance(embedding_i, embedding_j)
          pearson_dist = 1 - pearsonr(embedding_i, embedding_j)[0]
          cos_mean_dist += cos_dist
          p_mean_dist += pearson_dist
        cos_mean_dist /= len(embeddings)
        p_mean_dist /= len(embeddings)
        img_cos_mean_dist[fc_is_accurate] += cos_mean_dist
        img_p_mean_dist[fc_is_accurate] += p_mean_dist
      total_cos_mean_dist[fc_is_accurate] += (img_cos_mean_dist[fc_is_accurate]
                                              / len(embeddings))
      total_p_mean_dist[fc_is_accurate] += (img_p_mean_dist[fc_is_accurate]
                                            / len(embeddings))

      len_results += 1.0
    total_cos_mean_dist[0] /= (len_results-fc_accuracy)
    total_cos_mean_dist[1] /= fc_accuracy
    total_p_mean_dist[0] /= (len_results-fc_accuracy)
    total_p_mean_dist[1] /= fc_accuracy
    mlm_accuracy /= len_results
    fc_accuracy /= len_results


    print("Number of Results = " + str(len_results))
    print("Total MLM Accuracy = " + str(mlm_accuracy))
    print("Total Fake Context Accuracy = " + str(fc_accuracy))
    print("MLM Accuracy Ranking:")
    print(accuracy_vector)
    print("MLM Probabilities Ranking:")
    print(mlm_prob_vector)
    print("MLM Confusion Matrix:")
    print(mlm_conf_matrix.tolist())
    print("MLM Accuracy by Length:")
    print(mlm_length_accuracy)
    print("Fake Context Probabilities Ranking:")
    print(fc_prob_vector)
    print("Fake Context Confusion Matrix:")
    print(fc_conf_matrix)
    print("Fake Context Accuracy by Length:")
    print(fc_length_accuracy)
    print("Embeddings Mean Cos Distance by Accuracy:")
    print(total_cos_mean_dist)
    print("Embeddings Mean Pearson Distance by Accuracy:")
    print(total_p_mean_dist)
    print("Cluster Distance Probabilities Ranking:")
    print(dist_prob_vector)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  # flags.mark_flag_as_required("output_file")
  tf.app.run()
