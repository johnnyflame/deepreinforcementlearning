import gym
import numpy as np

# Theoretical source : http://karpathy.github.io/2016/05/31/rl/
# Practical source : https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0

"""
This section handles the observation pre_processing.

The basic steps are as follows:

1. Crop the image (we just care about the parts with information we care about.
2. Downsample the image.
3. Covert the image to black and white
4. Remove the background.
5. Covert from an 80 * 80 matrix of values to 6400 * 1 matrix (flatten the matrix so it's easier to use)
6. Store just the difference between the current frame and the previous fram if we know the previous frame(we only care about what's changed).

"""


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(vector):
    vector[vector < 0] = 0
    return vector


def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]


def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]


def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image


def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    """ See here: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop"""
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g ** 2
        weights[layer_name] += (learning_rate * g) / (np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name])  # reset batch gradient buffer


def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ covert the 210 * 160 * 3 uint8 frame into a 6400 float vector"""

    processed_observation = input_observation[35:195]  # crop image
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)

    processed_observation[processed_observation != 0] = 1  # everything else (paddles, ball) just set to 1


    # flatten the 80 * 80 matrix into 1 * 6400 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game.
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)

    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations


def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer
        values and the new output layer values """

    # we take the dot-product of the observation matrix and weight array no.1
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)

    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values


def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    """ See here: http://neuralnetworksanddeeplearning.com/chap2.html"""
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }


def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
        # signifies down in openai gym
        return 3


def discount_rewards(rewards, gamma):
    """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
    This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    """ discount the gradient with the normalized rewards """
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards




def main():
    env = gym.make("Pong-v0")
    observation = env.reset()  # this gets us the image.

    # hyperparameters

    batch_size = 10  # how many episodes to wait before adjusting weights
    gamma = 0.99  # discount factor for future reward
    decay_rate = 0.99  # Parameter used in RMSProp algorithm (Root mean squares propagation)
    num_hidden_layer_neurons = 200  # number of neurons
    input_dimensions = 80 * 80  # Dimension of our observation image
    learning_rate = 1e-4  # The step size in weight update.

    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    """
    We now set up weights

    Weights are stored in matrices.

    Layer 1 of our network is a 200 * 6400 matrix representing the weights for our hidden
    layer. For layer 1, element w1_ij represents the weight of neuron i for input pixel j in
    layer 1.

    Layer 2 is a 200 * 1 matrix representing the weights of the output of the hidden layer on
    our final output. For layer 2, element w2_i represents teh weights we place on the activation
    neural i in the hidden layer.

    We initialize each layer's weights with random numbers for now. We divide by the squareroot
    of the number of the dimension size to normalize our weights.
    """

    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }

    """
    parameters for RootMeanSqares propagation
    """

    # To be used with rmsprop algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)

    # Unsure how this works. Ask Lech

    expectation_g_squared = {}
    g_dict = {}

    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

    while True:
        env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation,
                                                                                      prev_processed_observations,
                                                                                      input_dimensions)
        """
           Now we send the observations through to our neural network to generate the probability of
           telling our AI to move up
        """

        hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)
        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        # carry out the chose action

        observation, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        # see here: http://cs231n.github.io/neural-networks-2/#losses
        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)


        if done:  # an episode finished
            episode_number += 1

            # Combine the following values for the episode, np.stack adds values in arrays
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            # Tweak the gradient of the log_ps based on the discounted rewards
            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)
            gradient = compute_gradient(
                episode_gradient_log_ps_discounted,
                episode_hidden_layer_values,
                episode_observations,
                weights
            )

            if episode_number % batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []  # reset values
            observation = env.reset()  # reset env
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
            reward_sum = 0
            prev_processed_observations = None


main()
