# Object Relation BERT

This project is an object context validator that adapts BERT to learn object relation representations by using Object Relation Modules in its self-attention layers.

Object Relation Modules (ORM) were introduced in Hu et. al 2018's paper [Relation Networks for Object Detection](https://arxiv.org/pdf/1711.11575.pdf). They are attention modules adapted to model the relation between objects in images based on their co-occurrence, positioning and scale.

Since the ORM only requires information about the objects' location in the image (bounding boxes), this context validator can be trained solely by information gathered from images' annotations. This allows us to detach the context modeling from the object detector, thus making possible to help any object detection algorithm in situations where the detection might be difficult, such as occlusion, poor image quality or adversarial examples.

In the original equation for the ORM, it's used an appearance feature extracted by a CNN from the object's region of interest in the image. This feature is used to represent the object. To avoid the need of image data, we replace this feature with an embedding created from the object's name. This way we have a representation for the object while only needing information present in the image's annotation.

Since we are using the objects' names as input, we will have a predefined vocabulary where each word represents an object, so the Word Piece tokenization used in BERT is not applied. The idea of learning the next sentence also does not apply to the Object Relation BERT, because a sentence will be composed by the names of each object present in an image, and the images might not be correlated. The positional embedding from BERT is also removed, since the order of objects' names in a sentence is irrelevant and their positional information will come from the ORM's geometric features.

The Object Relation BERT is trained on two binary tasks:

* An adapted Masked LM prediction, where instead of trying to predict the right word, the network will predict, for each object in the image, if the object belongs to the image's context or not.
* A fake context prediction, where it predicts whether the image's context as a whole makes sense. If the prediction is positive, it means there is probably at least one object that does not belong to this image's context.

Once trained, we can use the model to validate an object detector's results, identifying images where the context might be wrong and indicating the objects that may have been misclassified, helping to avoid such misclassifications.

## Runnning
Next we explain a little about the input data and show the parameters used when running this project in our experiments.

### Dataset
The input data is a text file where each line represent 1 image. The line contains the objects' names separated by space, followed by the separator '|', and then the objects' bounding boxes separated by space, with each coordinate from a single object separated by comma.

In the data directory there's two such files generated from COCO's 2017 training and validation dataset.

Since BERT has a limitation for the sequence length, it might be necessary to cut images contaning more objects than the sequence length. You can do that by truncating the image, or by generating many sequences from one image, permutating the objects.

### Create Data
```shell
python create_pretraining_data.py \
  --input_file=data/coco_train.bert \
  --output_file=data/tfrecords/coco_12_2.tfrecord \
  --vocab_file=data/coco_vocab.txt \
  --do_lower_case=True \
  --max_seq_length=12 \
  --max_predictions_per_seq=2 \
  --masked_lm_prob=0.15 \
  --fake_context_prob=0.5 \
  --random_seed=12345 \
  --dupe_factor=5
```

### Pre-Training
```shell
python run_pretraining.py \
  --input_file=data/tfrecords/coco_12_2.tfrecord \
  --output_dir=data/models/coco_12_2 \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=bert_config.json \
  --train_batch_size=45 \
  --max_seq_length=12 \
  --max_predictions_per_seq=2 \
  --num_train_steps=10000 \
  --num_warmup_steps=1000 \
  --learning_rate=2e-5 \
```

### Evaluation
```shell
python object_classifier.py \
  --input_file=data/coco_val.bert \
  --vocab_file=data/coco_vocab.txt \
  --bert_config_file=bert_config.json \
  --init_checkpoint=data/models/coco_12_2/model.ckpt-10000 \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=12 \
  --batch_size=67 \
  --do_eval=True \
  --do_predict=True \
  --fake_context_prob=0.5
```
