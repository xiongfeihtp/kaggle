from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ops import image_embedding
from ops import image_processing
from ops import inputs as input_ops

slim = tf.contrib.slim


class ShowAndTellModel(object):
    def __init__(self, config, mode, train_inception=False):
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.train_inception = train_inception

        # Reader for the input data.
        self.reader = tf.TFRecordReader()

        # To match the "Show and Tell" paper we initialize all variables with a
        # random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.images = None
        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # Collection of variables from the inception submodel.
        self.inception_variables = []

        # Function to restore the inception submodel from checkpoint.
        self.init_fn = None

        # Global step Tensor.
        self.global_step = None

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def process_image(self, encoded_image, thread_id=0):
        """Decodes and processes an image string.

        Args:
          encoded_image: A scalar string Tensor; the encoded image.
          thread_id: Preprocessing thread id used to select the ordering of color
            distortions.

        Returns:
          A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return image_processing.process_image(encoded_image,
                                              is_training=self.is_training(),
                                              height=self.config.image_height,
                                              width=self.config.image_width,
                                              thread_id=thread_id,
                                              image_format=self.config.image_format)

    def build_inputs(self):
        # Prefetch serialized SequenceExample protos.
        input_queue = input_ops.prefetch_input_data(
            self.reader,
            self.config.input_file_pattern,
            is_training=self.is_training(),
            batch_size=self.config.batch_size,
            values_per_shard=self.config.values_per_input_shard,
            # approximate values nums for all shard
            input_queue_capacity_factor=self.config.input_queue_capacity_factor,
            # queue_capacity_factor for shards
            num_reader_threads=self.config.num_input_reader_threads)

        # Image processing and random distortion. Split across multiple threads
        # with each thread applying a slightly different color distortions.
        assert self.config.num_preprocess_threads % 2 == 0
        images_and_label = []
        for thread_id in range(self.config.num_preprocess_threads):
            # thread
            serialized_sequence_example = input_queue.dequeue()
            encoded_image, image_label, image_name = input_ops.parse_sequence_example(
                serialized_sequence_example,
                image_feature=self.config.image_feature_name,
                label_feature=self.config.label_feature_name,
                filename_feature=self.config.filename_feature_name)
            # preprocessing, for different thread_id use different distortion function
            image = self.process_image(encoded_image, thread_id=thread_id)

            images_and_label.append([image, image_label,image_name])
            # mutil threads preprocessing the image


        queue_capacity = (2 * self.config.num_preprocess_threads *
                          self.config.batch_size)

        images, labels,image_names= tf.train.batch_join(
            images_and_label,
            batch_size=self.config.batch_size,
            capacity=queue_capacity,
            dynamic_pad=True,
            name="batch")

        
        self.images = images
        self.labels = labels
        self.image_names = image_names

    def build_image_embeddings(self):
        """Builds the image model subgraph and generates image embeddings.

        Inputs:
          self.images

        Outputs:
          self.image_embeddings
        """
        inception_output = image_embedding.inception_v3(
            self.images,
            trainable=self.train_inception,
            is_training=self.is_training())
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
        # Map inception output into embedding space.
        # embedding may be seen as a fully connected without activation_fn, biases, normalization
        with tf.variable_scope("image_embedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=inception_output,
                num_outputs=self.config.num_classes,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)
        # Save the embedding size in the graph.
        tf.constant(self.config.num_classes, name="num_classes")
        self.image_embeddings = image_embeddings

    def build_model(self):
        # one_hot_labels = slim.one_hot_encoding(self.labels,self.config.num_classes)
        # sparse_softmax don't need for one-hot labels
        predictions = tf.argmax(self.image_embeddings, 1)
        if self.mode != "inference":
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, self.labels)
            metrics_op = tf.group(accuracy_update,accuracy)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                    logits=self.image_embeddings)
            losses=tf.reduce_mean(losses)
            tf.losses.add_loss(losses)
            total_loss = tf.losses.get_total_loss()
            tf.summary.scalar("losses/total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)
            self.total_loss = total_loss
            self.metrics_op = metrics_op
        else:
            self.probabilities=tf.nn.softmax(self.image_embeddings,name='softmax')

    def setup_inception_initializer(self):
        """Sets up the function to restore inception variables from checkpoint."""
        if self.mode != "inference":
            # Restore inception variables only.
            saver = tf.train.Saver(self.inception_variables)

            def restore_fn(sess):
                tf.logging.info("Restoring Inception variables from checkpoint file %s",
                                self.config.inception_checkpoint_file)
                saver.restore(sess, self.config.inception_checkpoint_file)

            self.init_fn = restore_fn

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_image_embeddings()
        self.build_model()
        self.setup_inception_initializer()
        self.setup_global_step()
