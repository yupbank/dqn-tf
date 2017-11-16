import tensorflow as tf

NUM_OF_ACTION = 4
NUM_OF_FRAME_PER_STATE = 4
FRAME_SIZE = [80, 80]

def q_function(state, num_of_action=NUM_OF_ACTION, scope=''):
    with tf.variable_scope(scope):
        conv1 = tf.contrib.layers.conv2d(state, num_outputs=32, kernel_size=(8, 8), stride=(4, 4), scope='l1')
        conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=64, kernel_size=(4, 4), stride=(2, 2), scope='l2')
        conv3 = tf.contrib.layers.conv2d(conv2, num_outputs=64, kernel_size=(3, 3), stride=(1, 1), scope='l3')
        flattened = tf.contrib.layers.flatten(conv3, scope='flattened')
        fc = tf.contrib.layers.fully_connected(flattened, 512, scope='fc')
        score = tf.contrib.layers.fully_connected(fc, num_of_action, activation_fn=None, scope='action_score')
        return score


def _resize_into_gray(frame):
    gray_image = tf.image.rgb_to_grayscale(frame)
    return tf.image.resize_images(gray_image, FRAME_SIZE)

def _pack_frames_into_state(base, frame):
    old_dimesnion = base.shape.as_list()
    old_dimesnion[-1] = old_dimesnion[-1] - 1
    return tf.concat([tf.slice(base, [0, 0, 1], old_dimesnion), frame], axis=-1)

def frames_to_state(frames):
    frames_of_gray = tf.map_fn(_resize_into_gray, frames, dtype=tf.float32)
    first_frame = tf.squeeze(tf.slice(frames_of_gray, [0, 0, 0, 0], [1]+frames_of_gray.shape.as_list()[1:]))
    initializer = tf.concat([tf.expand_dims(first_frame, -1) for i in xrange(NUM_OF_FRAME_PER_STATE)], axis=-1)
    return tf.foldl(_pack_frames_into_state, frames_of_gray, initializer=initializer)


def prepare_action(action_holder, num_of_action=NUM_OF_ACTION):
    return tf.one_hot(action_holder, num_of_action, 1.0, 0.0, name='action_one_hot')


def action_score_to_action(action_score, epoch, epsilon_start, epsilon_end, epsilon_end_epoch):
    action_to_take = tf.argmax(action_score, axis=1)
    random_action = tf.cast(tf.random_uniform(shape=[1], minval=0.0, maxval=4.0), tf.int64)
    epsilon_start_with = (((epsilon_end - epsilon_start) / epsilon_end_epoch) * epoch + 1)
    epsilon = tf.where(epoch < epsilon_end_epoch, epsilon_start_with, epsilon_end)
    
    return tf.where(tf.random_uniform(shape=[1]) > epsilon, random_action, action_to_take)

def q_predicted_reward(action_score):
    return tf.reduce_max(action_score, axis=1)

def q_future_reward(action_score, action_holder, reward_holder, terminal_holder, discount):
    action_one_hot = prepare_action(action_holder)
    q_predicted = tf.reduce_sum(action_score * action_one_hot, axis=1)
    return reward_holder + (1.0 - terminal_holder) * discount * q_predicted

def loss(q_predicted_reward, q_truth_reward):
    return tf.losses.huber_loss(q_predicted_reward, q_truth_reward)
