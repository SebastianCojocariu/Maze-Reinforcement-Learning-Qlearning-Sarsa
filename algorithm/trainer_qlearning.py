from maze_qlearning import Maze
import random
from copy import copy
from random import choice, random
from argparse import ArgumentParser
from time import sleep
from matplotlib import pyplot as plt
import numpy as np
import json
from collections import deque
import math

PATH_HARTI = "harti/"

def exponential(value, theta):
	return value * theta

def RepresentsInt(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

def leeAlgorithm(matrix, i_start, j_start):
	n = len(matrix)
	m = len(matrix[0])
	if i_start < 0 or i_start >= n or j_start < 0 or j_start >= m:
		raise Exception("[LeeAlgorithm] i,j nu sunt coordonate valide, exiting...")
		exit(-1)
	
	if matrix[i_start][j_start] == '*':
		return None

	res = {(i_start, j_start): 0}
	queue = deque([(i_start, j_start)])
	while len(queue) > 0:
		(i, j) = queue.popleft()
		for (x, y) in ((-1, 0), (0, -1), (0, 1), (1, 0)):
			if i + x >= 0 and i + x < n and j + y >= 0 and j + y < m and matrix[i + x][j + y] != '*' and (i + x, j + y) not in res:
				res[(i + x, j + y)] = 1 + res[(i, j)]
				queue.append((i + x, j + y))
	return res

def calculateStatistics(Q):
	mean = 0
	std = 0
	n = 0
	minimum = float("Infinity")
	maximum = float("-Infinity")
	for key in Q:
		if Q[key] > maximum:
			maximum = Q[key]
		if minimum > Q[key]:
			minimum = Q[key]
		mean += Q[key]
		std += Q[key] * Q[key]
	mean = mean / len(Q)
	std = math.sqrt(std / len(Q) - mean * mean)

	return {"mean":mean, "std":std, "maximum":maximum, "minimum": minimum}


class Trainer:
	def __init__(self, map_file_name, policy_type, game_graphics, training_config, eval_config, game_hyperparameters,
				 compute_stats=False, eval_each_episode=False):
		# Args
		self.map_file_name = PATH_HARTI + map_file_name
		self.policy_type = policy_type
		self.game_graphics = game_graphics
		self.training_config = training_config
		self.eval_config = eval_config
		self.game_hyperparameters = game_hyperparameters
		self.compute_stats = compute_stats
		self.eval_each_episode = eval_each_episode

		# Q utility
		self.Q = {}

		# policy type
		self.policy_type = policy_type

		# hyperparameters initial values
		self.learning_rate = float(game_hyperparameters["learning_rate"]["initial_value"])
		self.epsilon = float(game_hyperparameters["epsilon"]["initial_value"])
		self.discount = float(game_hyperparameters["discount"]["initial_value"])

		# Game graphics
		self.final_show = game_graphics["final_show"].lower() == "true"
		self.verbose = game_graphics["verbose"].lower() == "true"
		self.sleep = float(game_graphics["sleep"])
		self.plot = game_graphics["plot"].lower() == "true"


		maze = Maze(self.map_file_name, self.epsilon)
		self.distances = {}

		for i in range(maze.n):
			for j in range(maze.m):
				self.distances[(i, j)] = leeAlgorithm(maze.matrix_map, i, j)



		if int(self.eval_config["frequency"]) > 0:
			self.eval_config["eval_every"] = int(self.training_config["training_episodes"]) // int(self.eval_config["frequency"])
		elif self.eval_config["eval_every"] != None:
			self.eval_config["eval_every"] = int(self.eval_config["eval_every"])


	def train(self):
		train_scores = []
		eval_scores = []
		list_percentage_won = []
		eval_each_episode_list = []
		utilities_statistics = {"mean":[], "std":[], "maximum":[], "minimum":[]}
		train_episodes = int(self.training_config["training_episodes"])

		print("---------------\nQlearning\n")
		print("Map: {}".format(self.map_file_name))
		print("Training episodes: {}".format(train_episodes))
		print("Policy: {}".format(self.policy_type))
		print("Learning_rate: {}, theta: {}".format(self.learning_rate, float(self.game_hyperparameters["learning_rate"]["theta"])))
		print("Epsilon: {}, theta: {}".format(self.epsilon, float(self.game_hyperparameters["epsilon"]["theta"])))
		print("Discount: {}, theta: {}".format(self.discount, float(self.game_hyperparameters["discount"]["theta"])))
		print("Verbose: {}".format(self.verbose))
		print("Sleep: {}\n\n---------------".format(self.sleep))

		additional_episodes, train_episode = 0, 0
		while (train_episode < train_episodes + additional_episodes):
			if train_episode % (train_episodes // 10) == 0:
				print("Completed {} % \n".format(int(train_episode // (train_episodes // 10)) * 10))
			# update hyperparameters
			if train_episode % int(self.game_hyperparameters["learning_rate"]["modify_every_no_episodes"]) == 0:
				aux = self.learning_rate
				theta_lr = float(self.game_hyperparameters["learning_rate"]["theta"])
				self.learning_rate = exponential(value=self.learning_rate, theta=theta_lr)
				print("Modified learning_rate from {} => {}".format(aux,self.learning_rate))
			
			if train_episode % int(self.game_hyperparameters["epsilon"]["modify_every_no_episodes"]) == 0:
				aux = self.epsilon
				theta_epsilon = float(self.game_hyperparameters["epsilon"]["theta"])
				self.epsilon = exponential(value=self.epsilon, theta=theta_epsilon)
				print("Modified epsilon from {} => {}".format(aux,self.epsilon))

			if train_episode % int(self.game_hyperparameters["discount"]["modify_every_no_episodes"]) == 0:
				aux = self.discount
				theta_discount = float(self.game_hyperparameters["discount"]["theta"])
				self.discount = exponential(value=self.discount, theta=theta_discount)
				print("Modified discount from {} => {}".format(aux,self.discount))


			maze = Maze(self.map_file_name, self.epsilon, distances=self.distances)


			# display current state and sleep
			if self.verbose:
				print("\nEpisode no {}\n---------------\nInitial Map:\n\n".format(train_episode))
				maze.print_map()
				sleep(self.sleep)


			# while current state is not terminal
			while not maze.is_final_state():

				# get curr_state
				curr_state = maze.get_serialized_current_state()

				# choose curr_action basen on policy type
				
				if self.policy_type.lower() == "maxfirst":
					curr_action = maze.maxFirst_policy(self.Q)

				elif self.policy_type.lower() == "random":
					curr_action = maze.random_policy(self.Q)

				elif self.policy_type.lower() == "exploatare":
					curr_action = maze.exploatare_policy(self.Q)

				elif self.policy_type.lower() == "explorare_exploatare":
					curr_action = maze.explorare_exploatare_policy(self.Q)

				else:
					raise Exception("uknown policy type. Exiting...")
					exit(-1)	


				# apply action and get the next state and the reward
				reward, msg = maze.apply_action(curr_action)
				
				# get curr_state after applying the proper action
				next_state = maze.get_serialized_current_state()
		

				# Q-Learning - update Q
				init_score = self.Q.get((curr_state, curr_action), 0.0)
				
				next_best_action = maze.best_action(self.Q)    

				max_reward = self.Q.get((next_state, next_best_action), 0)

				self.Q[(curr_state, curr_action)] = init_score + self.learning_rate * (reward + self.discount * max_reward - init_score)

				# display current state and sleep
				if self.verbose:
					print(msg)
					maze.print_map()
					print("")
					sleep(self.sleep)

			train_scores.append(maze.score)

			# evaluate the greedy policy


			if self.eval_each_episode:
				maze = Maze(self.map_file_name, self.epsilon, distances=self.distances)

				if self.verbose:
					print(msg)
					maze.print_map()
					print("")

				while not maze.is_final_state():

					action = maze.best_action(self.Q)
					reward, msg = maze.apply_action(action)

					if self.verbose:
						print(msg)
						maze.print_map()
						print("")

				eval_each_episode_list.append(maze.score)

			if train_episode % self.eval_config["eval_every"] == 0:
				avg_score = .0
				no_win = 0

				for _ in range(1, int(self.eval_config["number_of_simulated_games"]) + 1):

					maze = Maze(self.map_file_name, self.epsilon, distances=self.distances)

					if self.verbose:
							print(msg)
							maze.print_map()
							print("")

					while not maze.is_final_state():
						
						action = maze.best_action(self.Q)
						reward, msg = maze.apply_action(action)
						
						if self.verbose:
							print(msg)
							maze.print_map()
							print("")

					avg_score += maze.score

					if maze.winGame():
						no_win += 1

				if self.compute_stats:
					stats = calculateStatistics(self.Q)
					utilities_statistics["mean"].append(stats["mean"])
					utilities_statistics["std"].append(stats["std"])
					utilities_statistics["maximum"].append(stats["maximum"])
					utilities_statistics["minimum"].append(stats["minimum"])

				if int(self.eval_config["number_of_simulated_games"]) > 0:
					eval_scores.append((avg_score/int(self.eval_config["number_of_simulated_games"])))
					list_percentage_won.append(no_win / int(self.eval_config["number_of_simulated_games"]))
				else:
					eval_scores.append(float("-Infinity"))
					list_percentage_won.append(float("-Infinity"))

			train_episode += 1

			if self.final_show and train_episode == train_episodes + additional_episodes:
				while(True):
					aux = input("Da-mi numarul episoadelor aditionale\n")
					
					if RepresentsInt(aux):
						if int(aux) >= 0:
							additional_episodes += int(aux)
						break
					
					else:
						print("Numarul dat nu este valid, incearca din nou. Apasa 0 pentru a iesi din training\n")
						continue


		if int(self.eval_config["eval_final_wins"]) > 0:
			no_wins = 0
			for _ in range(int(self.eval_config["eval_final_wins"])):

				maze = Maze(self.map_file_name, self.epsilon, distances=self.distances)

				while not maze.is_final_state():

					action = maze.best_action(self.Q)
					reward, msg = maze.apply_action(action)


				if maze.winGame():
					no_wins += 1

			percentage_won = no_wins / int(self.eval_config["eval_final_wins"])
			#print(percentage_won)


		if self.final_show:
			while(True):
				print("\n\nFinal show:\nInitial Map: \n")
				maze = Maze(self.map_file_name, self.epsilon, distances=self.distances)

				# display current map
				maze.print_map()
				enter_press = True
				if enter_press:
					s = input("Press enter to continue step by step mode. Give z to resume demo or x to terminate\n")
					if s == 'z':
						enter_press = False
					elif s == 'x':
						break

				else:
					sleep(self.sleep)
				print("")
				final_score = 0
				while not maze.is_final_state():
					action = maze.best_action(self.Q)
					reward, msg = maze.apply_action(action)
					final_score += reward
					assert final_score == maze.score

					print(msg)
					maze.print_map()
					print("")

					if enter_press:
						s = input("Press enter to continue step by step mode. Give z to resume demo\n")
						if s == 'z':
							enter_press = False
					else:
						sleep(self.sleep)
				print("Final Score is {}\n".format(maze.score))
				s = input("Press any key to repeat the test or x to exit simulation.\n")
				if s == 'x':
					break


		if self.plot:
			n = train_episodes + additional_episodes
			plt.xlabel("Episode")
			plt.ylabel("Average score")
			plt.plot(
				np.linspace(1, train_episodes + additional_episodes, train_episodes + additional_episodes),
				np.convolve(train_scores, [0.2, 0.2, 0.2, 0.2, 0.2], "same"),
				#train_scores,
				linewidth=1.0, color="blue"
			)

			eval_every = self.eval_config["eval_every"]

			plt.plot(
				np.linspace(0, train_episodes + additional_episodes, len(eval_scores)),
				eval_scores, linewidth=2.0, color="red"
			)
			plt.show()

		res = {}
		res["train_scores"] = train_scores
		res["eval_scores"] = eval_scores
		res["eval_every"] = self.eval_config["eval_every"]
		res["learning_rate"] = self.learning_rate
		res["discount"] = self.discount
		res["percentage_won"] = percentage_won
		res["list_percentage_won"] = list_percentage_won
		res["utilities_stats"] = utilities_statistics
		res["eval_each_episode_list"] = eval_each_episode_list

		return res


with open('configs_qlearning.json') as json_file:
	configs = json.load(json_file)

map_file_name = configs["map_file_name"]
policy_type = configs["policy_type"]
game_graphics = configs["game_graphics"]
training_config = configs["training_config"]
eval_config = configs["eval_configs"]
game_hyperparameters = configs["game_hyperparameters"]

trainer = Trainer(map_file_name=map_file_name,
				  policy_type=policy_type,
				  game_graphics=game_graphics,
				  training_config=training_config,
				  eval_config=eval_config,
				  game_hyperparameters=game_hyperparameters
				 )

trainer.train()