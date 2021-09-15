from maze_qlearning import Maze
from maze_sarsa import Maze
import random
from copy import copy
from random import choice, random
from argparse import ArgumentParser
from time import sleep
from matplotlib import pyplot as plt
import numpy as np
import json
from collections import deque
from trainer_qlearning import Trainer
from trainer_sarsa import Trainer_Sarsa
import sys
from scipy.signal import lfilter

TASK1 = False
TASK2 = False
TASK3 = True
COMPARARE_SARSA_QLEARNING = False


if TASK1:
	print("TASK1")
	POLICY_TYPES = ["maxfirst", "random",  "exploatare", "explorare_exploatare"]
	PATH_TASK1 = "./grafice/task1"
	if len(sys.argv) < 2:
		raise Exception("give me file_name")
		exit(-1)

	map_file_name = sys.argv[1]
	configs_name = "configs_harti/configs_{}.json".format(map_file_name)

	with open(configs_name) as json_file:
		configs = json.load(json_file)

	# TASK 1
	map_file_name = configs["map_file_name"]
	game_graphics = configs["game_graphics"]
	training_config = configs["training_config"]
	eval_config = configs["eval_configs"]
	game_hyperparameters = configs["game_hyperparameters"]

	# TASK 1
	validation_list = []
	for policy_type in POLICY_TYPES:
		policy = policy_type

		trainer = Trainer(map_file_name=map_file_name,
					  policy_type=policy,
					  game_graphics=game_graphics,
					  training_config=training_config,
					  eval_config=eval_config,
					  game_hyperparameters=game_hyperparameters
					 )


		res = trainer.train()
		train_scores = res["train_scores"]
		eval_scores = res["eval_scores"]
		plt.title("[{}] {}".format(map_file_name, policy_type))
		validation_list.append([policy_type, len(train_scores), eval_scores])
		plt.xlabel("Episode")
		plt.ylabel("Average score")
		plt.plot(
			np.linspace(1, len(train_scores), len(train_scores)),
			np.convolve(train_scores, [0.2, 0.2, 0.2, 0.2, 0.2], "same"),
			linewidth=1.0,  label='Training', color="blue")


		plt.plot(
			np.linspace(1, len(train_scores), len(eval_scores)),
			eval_scores, linewidth=2.0, label='Validation', color="red")
		plt.legend(loc="best")
		plt.savefig("{}/{}_{}".format(PATH_TASK1, map_file_name, policy_type))
		plt.clf()

	COLORS = ["red", "green", "blue", "black"]

	for i, [policy_type, len_train, eval_scores] in enumerate(validation_list):
		plt.plot(
			np.linspace(1, len_train, len(eval_scores)),
			eval_scores, linewidth=1.0, label=policy_type, color=COLORS[i])

	plt.xlabel("Episode")
	plt.ylabel("Average score")
	plt.legend(loc="best")
	plt.savefig("{}/{}_comparison_all".format(PATH_TASK1, map_file_name))
	plt.clf()


# TASK 2
if TASK2:
	print("TASK2")
	POLICY_TYPES = ["explorare_exploatare" ,"random", "maxfirst", "exploatare"]
	PATH_TASK2 = "./grafice/task2"

	map_file_name = sys.argv[1]
	configs_name = "configs_harti/configs_{}.json".format(map_file_name)
	with open(configs_name) as json_file:
		configs = json.load(json_file)

	# TASK 2
	map_file_name = configs["map_file_name"]
	game_graphics = configs["game_graphics"]
	training_config = configs["training_config"]
	eval_config = configs["eval_configs"]
	game_hyperparameters = configs["game_hyperparameters"]

	len_train = int(training_config["training_episodes"])

	COLORS = ["red", "green", "blue", "black", "orange"]
	#LR
	for policy_type in POLICY_TYPES:
		dictionary = {}
		for lr in [0.1, 0.3, 0.5, 0.7, 0.9]:

			with open(configs_name) as json_file:
				configs = json.load(json_file)

			map_file_name = configs["map_file_name"]
			game_graphics = configs["game_graphics"]
			training_config = configs["training_config"]
			eval_config = configs["eval_configs"]
			game_hyperparameters = configs["game_hyperparameters"]

			game_hyperparameters["learning_rate"]["initial_value"] = lr
			eval_config["frequency"] = 100
			eval_config["number_of_simulated_games"] = 50

			trainer = Trainer(map_file_name=map_file_name,
							  policy_type=policy_type,
							  game_graphics=game_graphics,
							  training_config=training_config,
							  eval_config=eval_config,
							  game_hyperparameters=game_hyperparameters
							  )
			print("len_train = {}".format(len_train))
			res = trainer.train()
			dictionary[lr] = res["list_percentage_won"].copy()
			print("len(list_percentage) = {}".format(len(dictionary[lr])))


		for i, lr in enumerate(dictionary):
			if policy_type != "random":
				n = 14  # the larger n is, the smoother curve will be
				b = [1.0 / n] * n
				a = 1
				yy = lfilter(b, a, dictionary[lr])
				assert len(yy) == len(dictionary[lr])
				dictionary[lr] = yy
			else:
				pass
				n = 2  # the larger n is, the smoother curve will be
				b = [1.0 / n] * n
				a = 1
				yy = lfilter(b, a, dictionary[lr])
				assert len(yy) == len(dictionary[lr])
				dictionary[lr] = yy

			plt.plot(
				np.linspace(1, len_train, len(dictionary[lr])),
				dictionary[lr], linewidth=1.5, label="Lr = {}".format(lr), color=COLORS[i])
		plt.xlabel("Episode")
		plt.ylabel("Win ratio")
		plt.title("[{}] Learning Rate  {}".format(map_file_name, policy_type))
		plt.legend(loc="best")
		plt.savefig("{}/lr_{}_{}".format(PATH_TASK2, map_file_name, policy_type))
		plt.clf()

	#discount

	for policy_type in POLICY_TYPES:
		dictionary = {}
		for discount in [0.4, 0.55, 0.7, 0.85, 0.99]:
			with open(configs_name) as json_file:
				configs = json.load(json_file)

			map_file_name = configs["map_file_name"]
			game_graphics = configs["game_graphics"]
			training_config = configs["training_config"]
			eval_config = configs["eval_configs"]
			game_hyperparameters = configs["game_hyperparameters"]

			game_hyperparameters["discount"]["initial_value"] = discount
			eval_config["frequency"] = 100
			eval_config["number_of_simulated_games"] = 50

			trainer = Trainer(map_file_name=map_file_name,
							  policy_type=policy_type,
							  game_graphics=game_graphics,
							  training_config=training_config,
							  eval_config=eval_config,
							  game_hyperparameters=game_hyperparameters
							  )
			res = trainer.train()
			dictionary[discount] = res["list_percentage_won"].copy()

		for i, discount in enumerate(dictionary):
			if policy_type != "random":
				n = 14  # the larger n is, the smoother curve will be
				b = [1.0 / n] * n
				a = 1
				yy = lfilter(b, a, dictionary[discount])
				assert len(yy) == len(dictionary[discount])
				dictionary[discount] = yy
			else:
				pass
				n = 18  # the larger n is, the smoother curve will be
				b = [1.0 / n] * n
				a = 1
				yy = lfilter(b, a, dictionary[discount])
				assert len(yy) == len(dictionary[discount])
				dictionary[discount] = yy
			plt.plot(
				np.linspace(1, len_train, len(dictionary[discount])),
				dictionary[discount], linewidth=1.5, label="Discount = {}".format(discount), color=COLORS[i])
		plt.xlabel("Episode")
		plt.ylabel("Win ratio")
		plt.title("[{}] Discount  {}".format(map_file_name, policy_type))
		plt.legend(loc="best")
		plt.savefig("{}/discount_{}_{}".format(PATH_TASK2, map_file_name, policy_type))
		plt.clf()

if TASK3:
	print("TASK3")
	PATH_TASK3 = "./grafice/task3"
	dictionary = {"max_first":{"mean":{} , "minimum":{}, "maximum":{}, "std":{}},
				  "random":{"mean":{} , "minimum":{}, "maximum":{}, "std":{}}}

	len_train = 10000

	for map_file in ["harta_0", "harta_1", "harta_2", "harta_3"]:
		map_file_name = map_file
		configs_name = "configs_harti/configs_{}.json".format(map_file_name)
		with open(configs_name) as json_file:
			configs = json.load(json_file)

		# TASK 3
		map_file_name = configs["map_file_name"]
		game_graphics = configs["game_graphics"]
		training_config = configs["training_config"]
		training_config["training_episodes"] = len_train
		eval_config = configs["eval_configs"]
		game_hyperparameters = configs["game_hyperparameters"]

		# maxFirst
		policy_type = "maxFirst"
		trainer = Trainer(map_file_name=map_file_name,
						  policy_type=policy_type,
						  game_graphics=game_graphics,
						  training_config=training_config,
						  eval_config=eval_config,
						  game_hyperparameters=game_hyperparameters,
						  compute_stats=True,
						  )
		res_maxFirst = trainer.train()
		for key in res_maxFirst["utilities_stats"]:
			dictionary["max_first"][key][map_file_name] = res_maxFirst["utilities_stats"][key]

		# diferentele maxFirst random

		policy_type = "random"
		trainer = Trainer(map_file_name=map_file_name,
						  policy_type=policy_type,
						  game_graphics=game_graphics,
						  training_config=training_config,
						  eval_config=eval_config,
						  game_hyperparameters=game_hyperparameters,
						  compute_stats=True,
						  )
		res_random = trainer.train()
		for key in res_random["utilities_stats"]:
			dictionary["random"][key][map_file_name]  = res_random["utilities_stats"][key]


	# prima comparatie
	COLORS = ["red", "green", "blue", "black", "orange", "yellow", "pink", "purple", "magenta"]
	for key in dictionary["max_first"]:
		i = 0
		plt.title("MaxFirst {}".format(key))
		plt.xlabel("Episode")
		plt.ylabel("{}".format(key))
		for name_map in dictionary["max_first"][key]:
			aux_list = dictionary["max_first"][key][name_map]
			n = 7  # the larger n is, the smoother curve will be
			b = [1.0 / n] * n
			a = 1
			aux_list = lfilter(b, a, aux_list)

			plt.plot(
				np.linspace(1, len_train, len(aux_list)),
				aux_list,
				linewidth=1.5, label="{}".format(name_map), color=COLORS[i])
			i = (i + 1) % len(COLORS)
		plt.legend(loc="best")
		plt.savefig("{}/maxFirst_{}".format(PATH_TASK3, key))
		plt.clf()

	# a doua comparatie
	for key in dictionary["max_first"]:
		i = 0
		plt.title("MaxFirst vs Random {}".format(key))
		plt.xlabel("Episode")
		plt.ylabel("{}".format(key))
		for name_map in dictionary["max_first"][key]:
			aux_list_max_first = dictionary["max_first"][key][name_map]
			aux_list_random = dictionary["random"][key][name_map]
			n = 7  # the larger n is, the smoother curve will be
			b = [1.0 / n] * n
			a = 1
			aux_list_max_first = lfilter(b, a, aux_list_max_first)
			aux_list_random = lfilter(b, a, aux_list_random)
			plt.plot(
				np.linspace(1, len_train, len(aux_list_max_first)),
				aux_list_max_first,
				linewidth=1.0, label="{}_max_first".format(name_map), color=COLORS[i])
			i = (i + 1 )% len(COLORS)

			plt.plot(
				np.linspace(1, len_train, len(aux_list_random)),
				aux_list_random,
				linewidth=1.0, label="{}_random".format(name_map), color=COLORS[i])
			i = (i + 1) % len(COLORS)

		plt.legend(loc="best")
		plt.savefig("{}/comparison_{}".format(PATH_TASK3, key))
		plt.clf()



if COMPARARE_SARSA_QLEARNING:
	print("BONUS")
	POLICY_TYPES = ["exploatare", "maxfirst", "random", "explorare_exploatare"]
	PATH_BONUS = "./grafice/bonus"
	if len(sys.argv) < 2:
		raise Exception("give me file_name")
		exit(-1)

	map_file_name = sys.argv[1]

	if map_file_name != "harta_3":
		exit(0)

	configs_name = "configs_harti/configs_{}.json".format(map_file_name)

	with open(configs_name) as json_file:
		configs = json.load(json_file)

	map_file_name = configs["map_file_name"]
	game_graphics = configs["game_graphics"]
	training_config = configs["training_config"]
	eval_config = configs["eval_configs"]
	game_hyperparameters = configs["game_hyperparameters"]

	len_train = int(training_config["training_episodes"])

	for policy_type in POLICY_TYPES:
		trainer = Trainer(map_file_name=map_file_name,
						  policy_type=policy_type,
						  game_graphics=game_graphics,
						  training_config=training_config,
						  eval_config=eval_config,
						  game_hyperparameters=game_hyperparameters,
						  eval_each_episode=False
						  )

		res_qlearning = trainer.train()

		trainer = Trainer_Sarsa(map_file_name=map_file_name,
						  policy_type=policy_type,
						  game_graphics=game_graphics,
						  training_config=training_config,
						  eval_config=eval_config,
						  game_hyperparameters=game_hyperparameters,
						  eval_each_episode=False
						  )

		res_sarsa = trainer.train()

		eval_list_qlearning = res_qlearning["eval_scores"]
		eval_list_sarsa = res_sarsa["eval_scores"]
		n = 7  # the larger n is, the smoother curve will be
		b = [1.0 / n] * n
		a = 1
		eval_list_qlearning_filtered = lfilter(b, a, eval_list_qlearning)
		eval_list_sarsa_filtered = lfilter(b, a, eval_list_sarsa)
		plt.plot(
			np.linspace(1, len_train, len(eval_list_qlearning_filtered)),
			#np.convolve(eval_list_qlearning, [0.2, 0.2, 0.2, 0.2, 0.2], "same")[5:len_train - 5],
			eval_list_qlearning_filtered,
			linewidth=1, label="{}".format("Qlearning"), color="red")

		plt.plot(
			np.linspace(1, len_train - 5, len(eval_list_sarsa_filtered)),
			#np.convolve(eval_list_sarsa, [0.2, 0.2, 0.2, 0.2, 0.2], "same")[5:len_train - 5],
			eval_list_sarsa_filtered,
			linewidth=1, label="{}".format("Sarsa"), color="green")

		plt.title("[{}] Sarsa vs Qlearning for {}".format(map_file_name, policy_type))
		plt.xlabel("Episode")
		plt.ylabel("Reward")

		plt.legend(loc="best")
		plt.savefig("{}/{}_Sarsa_vs_Qlearning_{}".format(PATH_BONUS, map_file_name, policy_type))
		plt.clf()




