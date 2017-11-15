import model
import gym
import tensorflow as tf

def train():
    input_images = tf.placeholder(tf.float32, shape=[None, 210, 160, 3])
    action_holder = tf.placeholder(tf.int32, shape=[None])
    reward_input = tf.placeholder(tf.int32, shape=[None])
    terminal_mask = tf.placeholder(tf.int32, shape=[None])
    
    with tf.Graph().as_default() as g:
        input_state = model.prepare_imgae(input_images)
        action_one_hot = model.prepare_action(action_holder)
        action_score = model.q_function(input_state)

        q_target = tf.reduce_max(action_score, axis=1)

        q_acted = tf.reduce_sum(action_score * action_one_hot, axis=1)

        q_reward = reward_input + (1.0 - terminal_mask) * DISCOUNT * q_target
        
        loss = tf.losses.huber_loss(q_acted, q_reward)

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

        for episode in xrange(NUM_OF_EPISODE):
            for step in xrange(NUM_OF_STEP):
                if step % t == 0:
                    theta_data = sess.run(theta)

            
            random_action = tf.rint(tf.random_uniform([1], minval=0, maxval=NUM_OF_ACTION))
            
            random_number = tf.random_uniform([1])
           
            global_step = tf.Variable(0, trainable=False)

            starter_epsilon_rate = 0.1

            epsilon = tf.train.exponential_decay(epsilon, global_step,
                                                       100000, 0.96, staircase=True)

            next_action = tf.where(epsilon<random_number, random_action, q_target)

            action = sess.run(next_action, feed_dict={input_images: state_frame_t,
                                                      global_step: step+episode*NUM_OF_STEP})

            state_frame, reward, finished_episode, info = env.step(action)


            feed_dict = dict(zip(theta, theta_data))

            feed_dict.update({input_images_t1:state_frame_t1})

            q_selected_target_data = sess.run(q_target, feed_dict=feed_dict)

            sess.run(model_update, feed_dict={q_selected_target: q_selected_target_data, 
                                              input_images_t: input_images_t, 
                                              input_images_t1: input_images_t1, 
                                              action_holder: action_holder, 
                                              reward_input: reward, 
                                              terminal_mask: terminal})

if __name__ == "__main__":
    game_env = gym.make('BreakoutNoFrameskip-v4')
    observe = game_env.reset()
    print observe
    main()
