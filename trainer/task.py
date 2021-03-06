import model
import gym
import tensorflow as tf
import numpy as np
from collections import  deque
from operator import itemgetter

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/deep-q-network',
                                   """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_integer('NUM_OF_EPISODE', 1,
                                    """How many episode to run.""")
tf.app.flags.DEFINE_integer('NUM_OF_STEP', 100,
                                    """How many step to train in one episode.""")


tf.app.flags.DEFINE_float("EPSILON_START", 1, "Starting value for probability of greediness.")
tf.app.flags.DEFINE_float("EPSILON_END", 0.1, "Ending value for probability of greediness.")
tf.app.flags.DEFINE_float("EPSILON_END_EPOCH", 100, "Ending epoch to anneal epsilon.")

tf.app.flags.DEFINE_float("DISCOUNT", 0.99, "Amount to discount future rewards.")
tf.app.flags.DEFINE_integer("BURANOUT", 4, "Maximum Number of actions to play before every episode.")
tf.app.flags.DEFINE_integer("FRAME_PER_STATE", 4, "`agent history length` Number of frames of past history to model game state.")
tf.app.flags.DEFINE_integer("ACTION_SPACE", 4, "Number of possible output actions.")
tf.app.flags.DEFINE_integer("REPLAY_MEMORY_LENGTH", 1000000, "Number of historical experiences to store.")
tf.app.flags.DEFINE_integer("MIN_REPLAY_MEMORY_LENGTH", 50000, "Minimum number of experiences to start training.")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 32, "Size of mini-batch.")
tf.app.flags.DEFINE_integer("TARGET_NETWORK_UPDATE_FREQUENCY", 10000, "Rate at which to update the target network.")

FLAGS = tf.app.flags.FLAGS


def get_frame_buffer(maxlen=4):
    frame_buffer = deque(maxlen=maxlen)

    def append(frame):
        frame_buffer.append(frame)
        return frame_buffer

    return append


def sample(history, batch_size):
    history_size = len(history)
    index = np.random.randint(0, history_size, batch_size)
    sampled_memory = [history[i] for i in index]
    return (
            map(itemgetter(0), sampled_memory), 
            map(itemgetter(1), sampled_memory), 
            map(itemgetter(2), sampled_memory), 
            map(itemgetter(3), sampled_memory), 
            map(itemgetter(4), sampled_memory)
            )


def main(_):
    graph = tf.Graph()

    with graph.as_default():
        input_images = tf.placeholder_with_default(tf.zeros([1, 210, 160, 3], tf.float32), shape=[None, 210, 160, 3], name='input_images')
        action_holder = tf.placeholder(tf.int32, shape=[None], name='action_holder')
        reward_holder = tf.placeholder(tf.float32, shape=[None], name='reward_holder')
        terminal_holder = tf.placeholder(tf.float32, shape=[None], name='terminal_holder')
        epoch = tf.placeholder_with_default(tf.zeros([1], tf.float32), shape=[1], name='epoch')
        input_state = model.frames_to_state(input_images)
        _input_states = tf.expand_dims(input_state, 0)
        input_states = (
                            _input_states +
                            tf.placeholder_with_default(tf.zeros_like(_input_states), shape=[None, 80, 80, 4], name='batch_states')
                        )

        # util for play
        action_score = model.q_function(input_states)
        action_to_take = model.action_score_to_action(action_score, 
                                                      epoch, 
                                                      epsilon_start=FLAGS.EPSILON_START, 
                                                      epsilon_end=FLAGS.EPSILON_END, 
                                                      epsilon_end_epoch=FLAGS.EPSILON_END_EPOCH)

        # util for train 
        # the reason we expose future_reward is that we are using an old theta to calculate them
        q_future_reward = model.q_future_reward(action_score, 
                                                action_holder, 
                                                reward_holder,
                                                terminal_holder, 
                                                discount=FLAGS.DISCOUNT)
        q_predicted_reward = model.q_predicted_reward(action_score)
        loss = model.loss(q_predicted_reward, q_future_reward)
        trainer = tf.train.RMSPropOptimizer(
                learning_rate=0.00025,
                momentum=0.95
                )
        model_update = trainer.minimize(loss)

        theta = tf.trainable_variables()
        
        saver = tf.train.Saver(theta, max_to_keep=4)
    
    with tf.Session(graph=graph) as sess:
        game_env = gym.make('BreakoutNoFrameskip-v4')


        sess.run(tf.global_variables_initializer())

        append_frame = get_frame_buffer()

        memory_buffer = get_frame_buffer(maxlen=FLAGS.REPLAY_MEMORY_LENGTH)

         
        for episode in xrange(FLAGS.NUM_OF_EPISODE):
            observe = game_env.reset()
            for i in xrange(FLAGS.FRAME_PER_STATE):
                frames = append_frame(observe)

            for _ in xrange(FLAGS.BURANOUT):
                observe, reward, finished, _ = game_env.step(game_env.action_space.sample())
                frames.append(observe)
                if finished:
                    break

            prev_observe_state, action = sess.run([input_state, action_to_take], {input_images: frames})

            action = action[0]

            for t in xrange(FLAGS.NUM_OF_STEP):
                obs_frame, reward, finished, info = game_env.step(action)
                frames = append_frame(obs_frame)
                observe_state, next_action = sess.run([input_state, action_to_take], {input_images: frames, epoch: [episode]})
                next_action = next_action[0]
                history = memory_buffer([observe_state, reward, action, float(finished), prev_observe_state])
                prev_observe_state = observe_state
                action = next_action
                if finished:
                    game_env.reset()

                if t % FLAGS.TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                    theta_data = sess.run(theta)

                if len(history) > FLAGS.MIN_REPLAY_MEMORY_LENGTH:
                    states, rewards, actions, terminals, next_states = sample(history, FLAGS.BATCH_SIZE)

                    feed_dict = dict(zip(theta, theta_data))
                    feed_dict.update({
                                     input_states:next_states,
                                     terminal_holder: terminals,
                                     })

                    q_future_reward_data = sess.run(q_future_reward, feed_dict=feed_dict)

                    sess.run(model_update, feed_dict={
                                                     q_future_reward: q_future_reward_data, 
                                                     input_states:states,
                                                     terminal_holder: terminals,
                                                     action_holder: actions,
                                                     reward_holder: rewards, 
                                                      })
            saver.save(sess, FLAGS.checkpoint_dir, global_step=episode)

if __name__ == "__main__":
    tf.app.run()
