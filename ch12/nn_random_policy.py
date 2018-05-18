import tensorflow as tf
import numpy as np
import gym

env = gym.make("CartPole-v0")    

num_inputs = env.observation_space.shape[0]
inputs = tf.placeholder(tf.float32, shape=[None, num_inputs])
hidden = tf.layers.dense(inputs, 4, activation=tf.nn.relu)
outputs = tf.layers.dense(hidden, 1, activation=tf.nn.sigmoid)
action = tf.multinomial(tf.log(tf.concat([outputs, 1-outputs], 1)), 1)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	total_rewards = []
	for _ in range(1000):
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
