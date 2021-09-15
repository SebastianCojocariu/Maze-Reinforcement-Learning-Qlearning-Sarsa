from numpy.random import choice
import json
import random
from random import choice, random, choices
from scipy.special import softmax

with open('configs_sarsa.json') as json_file:
    configs = json.load(json_file)

game_rewards = configs["game_rewards"]

MOVE_REWARD = float(game_rewards["move_reward"])
WIN_REWARD = float(game_rewards["win_reward"])
LOSE_REWARD = float(game_rewards["lose_reward"])
CHEESE_REWARD = float(game_rewards["cheese_reward"])
MIN_SCORE_TRESHOLD = float(game_rewards["min_threshold"])

print("move_forward: {}".format(MOVE_REWARD))
print("cheese_reward: {}".format(CHEESE_REWARD))
print("win_reward: {}".format(WIN_REWARD))
print("lose_reward: {}".format(LOSE_REWARD))
print("MIN_SCORE_TRESHOLD: {}".format(MIN_SCORE_TRESHOLD))

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]
ACTION_EFFECTS = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "STAY": (0, 0)
}


class Maze:
    def __init__(self, map_file_name, epsilon, distances=None):

        self.read_from_file(map_file_name)
        self.epsilon = epsilon
        self.distances = distances
        self.legal_actions = {}
        for i in range(self.n):
            for j in range(self.m):
                aux = {}
                for action in ACTION_EFFECTS:
                    (x, y) = ACTION_EFFECTS[action]
                    if self.isValid(i + x, j + y):
                        aux[action] = (x, y)
                self.legal_actions[(i, j)] = aux

        self.score = 0
        self.mice_busted = False

        [m_row, m_column] = self.mice_current_coordinates
        if self.matrix_map[m_row][m_column] == "c":
            self.score += CHEESE_REWARD

        self.matrix_map[m_row][m_column] = 'J'

        c_row, c_column = self.cat_current_coordinates
        self.matrix_map[c_row][c_column] = 'T'

    def apply_action(self, action):
        curr_state = tuple(self.mice_current_coordinates)

        assert action in self.legal_actions[curr_state]
        message = "Jerry moved %s." % action

        # coordinates of Jerry
        [m_row, m_col] = self.mice_current_coordinates

        next_m_row = m_row + self.legal_actions[curr_state][action][0]
        next_m_col = m_col + self.legal_actions[curr_state][action][1]
        total_reward = 0

        self.mice_current_coordinates = [next_m_row, next_m_col]

        self.matrix_map[m_row][m_col] = " "

        # ran into TOM
        if self.matrix_map[next_m_row][next_m_col] == 'T':
            message = message + " Jerry ran into Tom. Jerry lost!:("
            self.mice_busted = True
            self.score += LOSE_REWARD
            return LOSE_REWARD, message

        # ran into a free cell
        elif self.matrix_map[next_m_row][next_m_col] == ' ':
            self.matrix_map[next_m_row][next_m_col] = 'J'

        # ran into a cell with cheese
        elif (next_m_row, next_m_col) in self.cheese_map:
            self.no_cheese -= 1
            del self.cheese_map[(next_m_row, next_m_col)]
            message = message + "Jerry collected a piece of cheese. Remaining pieces %d." % self.no_cheese
            self.matrix_map[next_m_row][next_m_col] = 'J'
            if self.no_cheese == 0:
                message = message + "Jerry won"
                self.score += (WIN_REWARD + CHEESE_REWARD)
                return CHEESE_REWARD + WIN_REWARD, message
            else:
                total_reward += CHEESE_REWARD

        [c_row, c_col] = self.cat_current_coordinates

        if (c_row, c_col) in self.cheese_map:
            self.matrix_map[c_row][c_col] = 'c'

        else:
            self.matrix_map[c_row][c_col] = ' '

        # if distance(Tom, Jerry) > self.A (Lee Algorithm) then Tom choose randomly a valid action
        if self.distances[(next_m_row, next_m_col)][(c_row, c_col)] > self.A:
            options = [action for action in self.legal_actions[(c_row, c_col)]]
            next_action = choice(options)
            (x, y) = self.legal_actions[(c_row, c_col)][next_action]
            [next_c_row, next_c_col] = [c_row + x, c_col + y]
            message = message + "Tom moves random towards %s." % next_action
        # Tom follows Jerry
        else:
            aux = []
            for action in self.legal_actions[(c_row, c_col)]:
                (i, j) = self.legal_actions[(c_row, c_col)][action]
                aux.append([action, self.distances[(next_m_row, next_m_col)][(c_row + i, c_col + j)]])

            aux.sort(key=lambda x: x[1])

            idx = 0
            while (idx < len(aux) and aux[idx][1] == aux[0][1]):
                idx += 1

            next_action = choice([action for (action, _) in aux[:idx]])
            (x, y) = self.legal_actions[(c_row, c_col)][next_action]
            [next_c_row, next_c_col] = [c_row + x, c_col + y]

            message = message + " Tom follows Jerry. Tom chose to move {}".format(next_action)

        if self.matrix_map[next_c_row][next_c_col] == "J":
            message = message + " Tom catches Jerry. Tom won!:("
            total_reward += LOSE_REWARD
            self.mice_busted = True

        else:
            total_reward += MOVE_REWARD

        self.cat_current_coordinates = [next_c_row, next_c_col]
        self.matrix_map[next_c_row][next_c_col] = "T"
        self.score += total_reward

        return total_reward, message

    def maxFirst_policy(self, Q):
        next_action = self.best_action(Q)

        return next_action

    def random_policy(self, Q):
        state = self.get_serialized_current_state()
        possible_actions = [action for action in self.legal_actions[state[:2]]]
        return choice(possible_actions)

    def exploatare_policy(self, Q):
        state = self.get_serialized_current_state()
        not_explored = [action for action in self.legal_actions[state[:2]] if not Q.get((state, action), None)]

        if len(not_explored) > 0:
            return choice(not_explored)

        elif random() < self.epsilon:
            all_possible_actions = [action for action in self.legal_actions[state[:2]]]
            return choice(all_possible_actions)

        return self.best_action(Q)

    def explorare_exploatare_policy(self, Q):
        state = self.get_serialized_current_state()

        all_possible_actions = [key for key in self.legal_actions[state[:2]]]

        '''
        eps = 0.1
        n = len(all_possible_actions)
        q_possible_actions = [Q.get((state, action), 0) for action in all_possible_actions]
        minimum = min(q_possible_actions)
        for i in range(len(q_possible_actions)):
            q_possible_actions[i] = q_possible_actions[i] - minimum + eps

        next_action = choices(all_possible_actions, weights=q_possible_actions)

        return next_action[0]
        '''
        q_possible_actions = [Q.get((state, action), 0) for action in all_possible_actions]
        maximum = max(q_possible_actions)
        for i in range(len(q_possible_actions)):
            q_possible_actions[i] = q_possible_actions[i] - maximum

        weights = softmax(q_possible_actions)

        next_action = choices(all_possible_actions, weights=weights)

        return next_action[0]

    def best_action(self, Q):
        state = self.get_serialized_current_state()
        possible_actions = [action for action in self.legal_actions[state[:2]]]
        q_possible_actions = [Q.get((state, action), 0) for action in possible_actions]
        max_reward = max(q_possible_actions)
        max_choices = []
        for i in range(len(q_possible_actions)):
            if q_possible_actions[i] == max_reward:
                max_choices.append(possible_actions[i])

        return choice(max_choices)

    def read_from_file(self, map_file_name):
        with open(map_file_name) as map_file:
            info = map_file.read().strip()

        info = info.split("\n")

        [self.n, self.m] = map(int, info[0].split())
        [self.A] = map(int, info[self.n + 1].split())
        self.mice_current_coordinates = list(map(int, info[self.n + 2].split()))
        self.cat_current_coordinates = list(map(int, info[self.n + 3].split()))

        info = list(map(list, info))

        ok = False
        if info[1][0] == '0' or info[1][0] == '1' or info[1][0] == '2':
            ok = True
            info = [[int(s) for s in line if s != ' '] for line in info]

        self.matrix_map = []
        for i in range(1, self.n + 1):
            if ok:
                for j in range(self.m):
                    if info[i][j] == 0:
                        info[i][j] = ' '
                    elif info[i][j] == 1:
                        info[i][j] = '*'
                    elif info[i][j] == 2:
                        info[i][j] = 'c'
                    else:
                        raise Exception("Matricea contine elemente diferite de 0, 1, 2")
                        exit(-1)

            self.matrix_map.append(info[i])

        self.no_cheese, self.cheese_map = 0, {}

        for i in range(self.n):
            for j in range(self.m):
                if self.matrix_map[i][j] == 'c':
                    self.cheese_map[(i, j)] = True
                    self.no_cheese += 1

        [x, y] = self.mice_current_coordinates
        self.matrix_map[x][y] = 'J'

        [x, y] = self.cat_current_coordinates
        self.matrix_map[x][y] = 'T'

        '''
        #transform to enunt type
        aux_matrix =[[None] * self.m for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.m):
                if self.matrix_map[i][j] == ' ':
                    aux_matrix[i][j] = '0'
                elif self.matrix_map[i][j] == "*":
                    aux_matrix[i][j] = '1'
                elif self.matrix_map[i][j] == 'c':
                    aux_matrix[i][j] = '2'
                elif self.matrix_map[i][j] == 'T' or self.matrix_map[i][j] == 'J':
                    aux_matrix[i][j] = '0'
                else:
                    raise Exception("Matricea contine elemente diferite de 0, 1, 2")
                    exit(-1)
        for x in aux_matrix:
            print(" ".join(x))

        exit(-1)
        '''

        '''
        print(self.n, self.m)
        for x in self.matrix_map:
            print(x)
        print(self.A)
        print(self.mice_current_coordinates)
        print(self.cat_current_coordinates)
        for x in self.cheese_map:
            print(x)
        '''

    def get_serialized_current_state(self):
        return tuple(self.mice_current_coordinates) + tuple(self.cat_current_coordinates) + tuple(
            sorted(self.cheese_map))

    def winGame(self):
        return (self.no_cheese == 0)

    def is_final_state(self):
        return (self.score < MIN_SCORE_TRESHOLD or self.mice_busted or self.no_cheese == 0)

    def isValid(self, row, column):
        return (row >= 0 and row < self.n and column >= 0 \
                and column < self.m and self.matrix_map[row][column] != "*")

    def print_map(self):
        print("\n".join(map(lambda row: "".join(row), self.matrix_map)))

# def deserialize_state(self):
#	return list(map(list, self.matrix_map.split("\n")))





