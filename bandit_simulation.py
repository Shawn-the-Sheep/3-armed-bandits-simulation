import numpy as np
import random

class RewardGenerator():

    def __init__(self, prob, mean = 5, std_dev = 1.5):

        self.mean = mean
        self.std_dev = std_dev
        self.probability_of_change = prob
    
    def get_mean(self):

        return self.mean

    def set_prob(self, new_prob):

        self.probability_of_change = new_prob

    def potentially_change_mean(self):

        if random.random() < self.probability_of_change:

            self.mean += 1

    def pull_slot(self):

        return round(np.random.normal(loc = self.mean, scale = self.std_dev))

class Agent():

    def __init__(self):

        self.score = 0
    
    def get_score(self):

        return self.score

    def increment_score(self, reward):

        self.score += reward

    def choose_slot(self):

        valid_choices = [str(i) for i in range(1, 4)]

        valid_choices.extend(["Q", "q"])

        valid_choice = False

        choice = -1 #Placeholder

        while not valid_choice:

            choice = input("Enter which reward generator you would like to receive a reward from (1-3 or Q/q to quit): ")

            if choice in valid_choices:

                valid_choice = True

            else:
                
                print("\nWhat you entered is not valid according to the allowed values\n")

        if choice.isdigit():

            return int(choice) - 1

        else:

            return choice
    
    def __str__(self):

        return f"player score: {self.score}"

class Model(Agent):

    def __init__(self, mean_slot1, mean_slot2, mean_slot3, learning_rate = 0.4, epsilon = 0.3):
        
        #we are using the epsilon-greedy policy approach where there is an epsilon probability each turn for the model to choose a random slot to
        #encourage exploration 
        
        super().__init__()
        self.estimated_expected_reward_slot1 = mean_slot1
        self.estimated_expected_reward_slot2 = mean_slot2
        self.estimated_expected_reward_slot3 = mean_slot3
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def train(self, k, reward):

        if k == 1:

            self.estimated_expected_reward_slot1 += self.learning_rate * (reward - self.estimated_expected_reward_slot1)

        elif k == 2:

            self.estimated_expected_reward_slot2 += self.learning_rate * (reward - self.estimated_expected_reward_slot2)

        elif k == 3:

            self.estimated_expected_reward_slot3 += self.learning_rate * (reward - self.estimated_expected_reward_slot3)
    
    def choose_slot(self):

        if random.random() < self.epsilon:

            return random.randint(0, 2)

        elif self.estimated_expected_reward_slot1 >= self.estimated_expected_reward_slot2 and self.estimated_expected_reward_slot1 >= self.estimated_expected_reward_slot3:

            return 0

        elif self.estimated_expected_reward_slot2 >= self.estimated_expected_reward_slot1 and self.estimated_expected_reward_slot2 >= self.estimated_expected_reward_slot3:

            return 1

        elif self.estimated_expected_reward_slot3 >= self.estimated_expected_reward_slot1 and self.estimated_expected_reward_slot3 >= self.estimated_expected_reward_slot2:

            return 2
    
    def __str__(self):

        return f"model score: {self.get_score()}"
    
def main_game():

    min_prob = 0.05     #min prob for a reward generator to change its mean
    max_prob = 0.2      #analogous to min prob conceptually but in the max case

    game_intro = '''Welcome to the non-stationary 3-armed bandit demonstration

    We have 3 reward generators if you will that will generate for you a reward if you pull it's lever. The key thing to remember
    is that each reward generator starts off with a normal distribution with mean = 5 and std_dev = 1.5 as its probability
    distribution and this probability distribution is also not constant and is subject to change. You will be competing with a model
    that uses a simple Reinforcement Learning Algorithm to attempt to maximize its rewards and after each turn by you and the model,
    there will be a chance that any of the distributions will be shifted to the right. The 3 reward generators will be numbered
    from 1-3 and you will have to choose 1 of them to pull in order to receive your reward for that round.
    '''

    prob_change1 = random.uniform(min_prob, max_prob)
    prob_change2 = random.uniform(min_prob, max_prob)
    prob_change3 = random.uniform(min_prob, max_prob)
    
    first_reward_generator = RewardGenerator(prob = prob_change1)
    second_reward_generator = RewardGenerator(prob = prob_change2)
    third_reward_generator = RewardGenerator(prob = prob_change3)

    reward_generators = [first_reward_generator, second_reward_generator, third_reward_generator]

    player = Agent()
    model = Model(mean_slot1 = first_reward_generator.get_mean(), mean_slot2 = second_reward_generator.get_mean(), mean_slot3 = third_reward_generator.get_mean())

    round_num = 1

    print(game_intro + "\n")

    while True:

        print(f"Round {round_num}: \n")

        if round_num % 100 == 0:

            new_prob_change1 = random.uniform(min_prob, max_prob)
            new_prob_change2 = random.uniform(min_prob, max_prob)
            new_prob_change3 = random.uniform(min_prob, max_prob)

            first_reward_generator.set_prob(new_prob_change1)
            second_reward_generator.set_prob(new_prob_change2)
            third_reward_generator.set_prob(new_prob_change3)
        
        player_choice = player.choose_slot()

        if player_choice == "Q" or player_choice == "q":

            print("player has decided to end the game")
            break

        curr_player_reward = reward_generators[player_choice].pull_slot()

        player.increment_score(curr_player_reward)

        model_choice = model.choose_slot()

        curr_model_reward = reward_generators[model_choice].pull_slot()

        model.increment_score(curr_model_reward)

        model.train(model_choice + 1, curr_model_reward)

        print("\n")
        print(player)
        print(model)
        print("\n")

        for reward_generator in reward_generators:

            reward_generator.potentially_change_mean()
        
        round_num += 1

main_game()