import tensorflow as tf
import numpy as np
import gym

env = gym.make("CartPole-v0")

num_inputs = env.observation_space.shape[0]
inputs = tf.placeholder(tf.float32, shape=[None, num_inputs])
y = tf.placeholder(tf.float32, shape=[None, 1])
hidden = tf.layers.dense(inputs, 4, activation=tf.nn.relu)

logits = tf.layers.dense(hidden, 1)
outputs = tf.nn.sigmoid(logits)

#outputs = tf.layers.dense(hidden, 1, activation=tf.nn.sigmoid)
action = tf.multinomial(tf.log(tf.concat([outputs, 1-outputs], 1)), 1)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(0.01)
training_op = optimizer.minimize(cross_entropy)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# training
	for _ in range(1000):
		obs = env.reset()

		while True:
			y_target = np.array([[1. if obs[2] < 0 else 0.]])
			a, _ = sess.run([action, training_op], feed_dict={inputs: obs.reshape(1, num_inputs), y: y_target})
			obs, reward, done, info = env.step(a[0][0])
			if done:
				break
	print("training done")

	# test
	total_rewards = []
	for _ in range(1000):
		rewards = 0
		obs = env.reset()

		while True:
			y_target = np.array([1. if obs[2] < 0 else 0.])
			a = sess.run(action, feed_dict={inputs: obs.reshape(1, num_inputs)})
			obs, reward, done, info = env.step(a[0][0])
			rewards += reward
			if done:
				break
		total_rewards.append(rewards)

	print(np.mean(total_rewards))

# 42.321
# 44.253
