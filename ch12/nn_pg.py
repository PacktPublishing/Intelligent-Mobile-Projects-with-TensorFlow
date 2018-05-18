import tensorflow as tf
import numpy as np
import gym

def normalized_discounted_rewards(rewards):
    dr = np.zeros(len(rewards))
    dr[-1] = rewards[-1]
    for n in range(2, len(rewards)+1):
        dr[-n] = rewards[-n] + dr[-n+1] * discount_rate
    return (dr - dr.mean()) / dr.std() # dr

# def ndrewards(all_rewards):
#     drs = [drewards(rewards) for rewards in all_rewards]
#     return [(dr - np.concatenate(drs).mean())/np.concatenate(drs).std() for dr in drs]

env = gym.make("CartPole-v0")

learning_rate = 0.05
discount_rate = 0.95

num_inputs = env.observation_space.shape[0]
inputs = tf.placeholder(tf.float32, shape=[None, num_inputs])
hidden = tf.layers.dense(inputs, 4, activation=tf.nn.relu) 
logits = tf.layers.dense(hidden, 1)
outputs = tf.nn.sigmoid(logits)  
action = tf.multinomial(tf.log(tf.concat([outputs, 1-outputs], 1)), 1)

prob_action_0 = tf.to_float(1-action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=prob_action_0)
optimizer = tf.train.AdamOptimizer(learning_rate)

gvs = optimizer.compute_gradients(cross_entropy)
gvs = [(g, v) for g, v in gvs if g != None]
gs = [g for g, _ in gvs]

gps = []
gvs_feed = []
for g, v in gvs:
    gp = tf.placeholder(tf.float32, shape=g.get_shape())
    gps.append(gp)
    gvs_feed.append((gp, v))
training_op = optimizer.apply_gradients(gvs_feed)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # play and train
    for n in range(1000): # 200 iterations
        print(str(n))

        # play a complete game first
        rewards, grads = [], []
        obs = env.reset()
        while True:
            a, gs_val = sess.run([action, gs], feed_dict={inputs: obs.reshape(1, num_inputs)})
            obs, reward, done, info = env.step(a[0][0])
            rewards.append(reward)
            grads.append(gs_val)
            if done:
                break

        # update gradients and do the training 
        nd_rewards = normalized_discounted_rewards(rewards)
        gp_val = {}
        for i, gp in enumerate(gps):
            gp_val[gp] = np.mean([grads[k][i] * reward
                                      for k, reward in enumerate(nd_rewards)], axis=0)
        sess.run(training_op, feed_dict=gp_val)
	
    #saver.save(sess, "./nnpg.ckpt")    
    # tf.train.write_graph(sess.graph_def, "model", 'nnpg.pbtxt')
    # tf.summary.FileWriter("logdir", sess.graph_def)

    print("play and train done")

    # test with the trained model
    total_rewards = []

    for _ in range(100):
    	rewards = 0
    	obs = env.reset()

    	while True:
    		a = sess.run(action, feed_dict={inputs: obs.reshape(1, num_inputs)})
    		obs, reward, done, info = env.step(a[0][0])
    		rewards += reward
    		if done:
    			break
    	total_rewards.append(rewards)

    print(np.mean(total_rewards))




# test with the trained model
# total_rewards = []

# with tf.Session() as sess:
#     saver.restore(sess, "./nnpg.ckpt")
#     for _ in range(1000):
#     	rewards = 0
#     	obs = env.reset()

#     	while True:
#     		a = sess.run(action, feed_dict={inputs: obs.reshape(1, num_inputs)})
#     		obs, reward, done, info = env.step(a[0][0])
#     		rewards += reward
#     		if done:
#     			break
#     	total_rewards.append(rewards)

#     print(np.mean(total_rewards))

# train with 250 iterations with 0.01 learning rate
# 195.851
# 197.077
# 196.538

# train with 500 iterations with 0.01 learning rate
# 199.558
# 199.541

# 200 iterations with 0.05 learning rate
# 199.9

# 200 iterations with 0.05 learning rate 0.8 discount rate
# 137


