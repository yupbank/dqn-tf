import model
import gym
import tensorflow as tf
import numpy as np
from collections import  deque


tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/inception_train',
                                   """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', '/tmp/inception_output',
                                   """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('NUM_OF_EPISODE', 1,
                                    """How many episode to run.""")
tf.app.flags.DEFINE_integer('NUM_OF_STEP', 100,
                                    """How many step to train in one episode.""")


tf.app.flags.DEFINE_float("EPSILON_START", 1, "Starting value for probability of greediness.")
tf.app.flags.DEFINE_float("EPSILON_END", 0.1, "Ending value for probability of greediness.")
tf.app.flags.DEFINE_float("EPSILON_END_EPOCH", 1000000, "Ending epoch to anneal epsilon.")

tf.app.flags.DEFINE_float("DISCOUNT", 0.99, "Amount to discount future rewards.")
tf.app.flags.DEFINE_integer("BURANOUT", 4, "Number of frames to play before training a batch.")
tf.app.flags.DEFINE_integer("FRAME_PER_STATE", 4, "Number of frames of past history to model game state.")
tf.app.flags.DEFINE_integer("ACTION_SPACE", 4, "Number of possible output actions.")
tf.app.flags.DEFINE_integer("REPLAY_MEMORY_LENGTH", 500000, "Number of historical experiences to store.")
tf.app.flags.DEFINE_integer("MIN_REPLAY_MEMORY_LENGTH", 50000, "Minimum number of experiences to start training.")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 32, "Size of mini-batch.")
tf.app.flags.DEFINE_integer("TARGET_NETWORK_UPDATE_FREQUENCY", 10000, "Rate at which to update the target network.")

FLAGS = tf.app.flags.FLAGS


def epsilon(epoch):
        return (((FLAGS.EPSILON_END - FLAGS.EPSILON_START) / FLAGS.EPSILON_END_EPOCH) * epoch + 1) if epoch < FLAGS.EPSILON_END_EPOCH else FLAGS.EPSILON_END


def get_frame_buffer(frame, max_len=4, fill=True):
    frame_buffer = deque(max_len=max_len)

    def append(frame):
        return frame_buffer.append(frame)

    if fill:
        for i in xrange(max_len):
            append(frame)

    return frame_buffer, append



def main(_):
    input_images = tf.placeholder(tf.float32, shape=[None, 210, 160, 3])
    action_holder = tf.placeholder(tf.int32, shape=[None])
    reward_input = tf.placeholder(tf.int32, shape=[None])
    terminal_holder = tf.placeholder(tf.float32, shape=[None])
    epoch = tf.placeholde(tf.float32, shape=[1])
    
    with tf.Graph().as_default() as g:
        # util for play
        input_state = model.frames_to_state(input_images)
        input_states = tf.expand_dims(input_states, 0)

        action_score = model.q_function(input_states)
        action_to_take = model.action_score_to_action(action_score)

        # util for train 
        # the reason we expose future_reward is that we are using an old theta to calculate them
        q_future_reward = model.q_future_reward(action_score, 
                                                action_holder, 
                                                terminal_holder, 
                                                discount=FLAGS.DISCOUNT)
        q_predicted_reward = model.q_predicted_reward(action_score)

        loss = model.loss(q_predicted_reward, q_truth_reward)

        trainer = tf.train.RMSPropOptimizer(
                learning_rate=0.00025,
                momentum=0.95
                )

        model_update = trainer.minimize(loss)

        theta = tf.trainable_variables()
    
    with tf.Session() as sess, g.as_default():
        game_env = gym.make('BreakoutNoFrameskip-v4')

        observe = game_env.reset()

        sess.run(tf.global_variables_initializer())

        frames, append_frame = get_frame_buffer(observe)
        
        prev_observe_state, action = sess.run([input_state, action], {input_images: frames})

         
        for episode in xrange(FLAGS.NUM_OF_EPISODE):
            history = []
            for step in xrange(FLAGS.REPLAY_MEMORY_LENGTH):
                obs_frame, reward, finished_episode, info = game_env.step(next_action)
                frames = append_frame(obs_frame)
                observe_state, next_action = sess.run([input_state, action], {input_images: frames})
                history.append([observe_state, reward, action, float(finished_episode), prev_observe_state])
                prev_observe_state = observe_state
                action = next_action
            
            for step in xrange(FLAGS.NUM_OF_STEP):
                states, rewards, actions, terminals, next_states = sample(history, batch_size)
                if step % FLAGS.TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                    theta_data = sess.run(theta)
                feed_dict = dict(zip(theta, theta_data))
                feed_dict.update({input_state:states})

                q_future_reward_data = sess.run(q_future_reward, feed_dict=feed_dict)

                sess.run(model_update, feed_dict={q_future_reward: q_future_reward_data, 
                                                  input_state: next_states, 
                                                  action_holder: actions,
                                                  reward_input: reward, 
                                                  terminal_holder: terminal})

if __name__ == "__main__":
    tf.app.run()
