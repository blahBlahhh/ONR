import ONR.utils as utils

import tensorflow as tf

import time
import os


class Convnet:
    def __init__(self, batch_size=128, learning_rate=0.001):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.skip_step = 20
        self.n_test = 10000

    def _get_data(self):
        """Grabs data from MNIST dataset"""
        with tf.name_scope('data'):
            train_data, test_data = utils.get_mnist_dataset(self.batch_size)
            self.iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                            train_data.output_shapes)
            img, self.label = self.iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])

            self.train_init = self.iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = self.iterator.make_initializer(test_data)  # initializer for train_data

    def conv_relu(self, inputs, filters, k_size, stride, padding, scope_name):
        """Creates convolution layer and relu activation layer"""
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            in_channels = inputs.shape[-1]
            kernel = tf.get_variable('kernel',
                                     [k_size, k_size, in_channels, filters],
                                     initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable('biases',
                                     [filters],
                                     initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
        return tf.nn.relu(conv + biases, name=scope.name)

    def maxpool(self, inputs, k_size, stride, padding='VALID', scope_name='pool'):
        """Creates max pooling layer"""
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            pool = tf.nn.max_pool(inputs,
                                  ksize=[1, k_size, k_size, 1],
                                  strides=[1, stride, stride, 1],
                                  padding=padding)
        return pool

    def fully_connected(self, inputs, out_dim, scope_name='fc'):
        """Creates fully connected layer"""
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            in_dim = inputs.shape[-1]
            w = tf.get_variable('weights', [in_dim, out_dim],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [out_dim],
                                initializer=tf.constant_initializer(0.0))
            out = tf.matmul(inputs, w) + b
        return out

    def _create_model(self):
        """Creates CNN model"""
        conv1 = self.conv_relu(inputs=self.img,
                               filters=32,
                               k_size=5,
                               stride=1,
                               padding='SAME',
                               scope_name='conv1')
        pool1 = self.maxpool(conv1, 2, 2, 'VALID', 'pool1')
        conv2 = self.conv_relu(inputs=pool1,
                               filters=64,
                               k_size=5,
                               stride=1,
                               padding='SAME',
                               scope_name='conv2')
        pool2 = self.maxpool(conv2, 2, 2, 'VALID', 'pool2')
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = self.fully_connected(pool2, 1024, 'fc')
        dropout = tf.nn.dropout(tf.nn.relu(fc), 0.75, name='relu_dropout')
        self.logits = self.fully_connected(dropout, 10, scope_name='logits')

    def _create_loss(self):
        """Creates cross entropy loss"""
        with tf.name_scope("loss"):
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name="loss")

    def _create_optimizer(self):
        """Creates AdamOptimizer"""
        self.gstep = tf.get_variable(name='global_step', dtype=tf.int32,
                                     initializer=tf.constant(0), trainable=False)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

    def _create_summaries(self):
        """Creates summary of loss and accuracy"""
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def evaluate(self):
        """Counts the number of right predictions in a batch"""
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        """Builds the whole model"""
        self._get_data()
        self._create_model()
        self._create_loss()
        self._create_optimizer()
        self.evaluate()
        self._create_summaries()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        """Trains data for one epoch (as the input iterator is out of range)"""
        start_time = time.time()
        sess.run(init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, '../checkpoints/', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        """Tests data after training an epoch (as the input iterator is out of range)"""
        start_time = time.time()
        sess.run(init)
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        """Trains the model multiple times"""
        utils.safe_mkdir('../graphs/convnet')
        writer = tf.summary.FileWriter('../graphs/convnet', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            utils.safe_mkdir('../checkpoints/')
            ckpt = tf.train.get_checkpoint_state('../checkpoints/')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)

            saver.save(sess, '../checkpoints/final_model', step)
        writer.close()

    def predict(self, sess, img):
        """Predicts the input image"""
        self.img = tf.reshape(tf.cast(img, tf.float32), shape=[1, 28, 28, 1])
        self._create_model()

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('../checkpoints/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        pred = tf.nn.softmax(self.logits)
        prediction = tf.argmax(pred, 1)
        logit, result = sess.run((pred, prediction))
        print(logit)
        print(result)

        return result


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model = Convnet()
    model.build()
    model.train(n_epochs=30)
