import streamlit as st
import numpy as np
import random

class RewardGenerator:

    """Represents a single slot machine (a 'bandit')."""

    def __init__(self, prob, mean=5, std_dev=1.5):

        self.mean = mean
        self.std_dev = std_dev
        self.probability_of_change = prob

    def get_mean(self):

        return self.mean

    def set_prob(self, new_prob):

        self.probability_of_change = new_prob

    def potentially_change_mean(self):

        """Simulates non-stationarity by potentially increasing the mean."""

        if random.random() < self.probability_of_change:
            self.mean += 1

    def pull_slot(self):

        """Generates a reward based on the current normal distribution."""

        return round(np.random.normal(loc=self.mean, scale=self.std_dev))

class Agent:

    """A base class for any entity that has a score."""

    def __init__(self):
        self.score = 0

    def get_score(self):
        return self.score

    def increment_score(self, reward):
        self.score += reward

class Model(Agent):

    """The RL Agent that learns to play the game."""

    def __init__(self, num_slots, learning_rate=0.4, epsilon=0.3):

        super().__init__()
        self.q_values = [5.0] * num_slots  # Initialize with the starting mean and better to use a list as it would be easier to add slots if we want
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_slots = num_slots

    def train(self, slot_index, reward):

        """Updates the Q-value for the chosen slot using the learning formula."""

        old_q = self.q_values[slot_index]
        self.q_values[slot_index] = old_q + self.learning_rate * (reward - old_q)

    def choose_slot(self):

        """Chooses a slot using the epsilon-greedy policy."""

        if random.random() < self.epsilon:
            return random.randint(0, self.num_slots - 1)
        else:
            max_q = max(self.q_values)
            best_actions = [i for i, q in enumerate(self.q_values) if q == max_q]
            return random.choice(best_actions)

def initialize_game():

    """Sets up the initial state of the game and stores it in the session."""

    min_prob, max_prob = 0.05, 0.2
    num_slots = 3
    
    st.session_state.reward_generators = [
        RewardGenerator(prob=random.uniform(min_prob, max_prob)) for _ in range(num_slots)
    ]
    st.session_state.player = Agent()
    st.session_state.model = Model(num_slots=num_slots)
    
    st.session_state.player_last_reward = 0
    st.session_state.model_last_reward = 0
    st.session_state.model_last_choice = None
    st.session_state.cheat_mode = False


def play_turn(player_choice_index):

    """Executes a full turn for both the player and the model."""

    player_reward = st.session_state.reward_generators[player_choice_index].pull_slot()
    st.session_state.player.increment_score(player_reward)
    st.session_state.player_last_reward = player_reward
    
    model_choice_index = st.session_state.model.choose_slot()
    st.session_state.model_last_choice = model_choice_index
    model_reward = st.session_state.reward_generators[model_choice_index].pull_slot()
    st.session_state.model.increment_score(model_reward)
    st.session_state.model_last_reward = model_reward

    old_q = st.session_state.model.q_values[model_choice_index]
    new_q = old_q + st.session_state.model.learning_rate * (model_reward - old_q)

    st.session_state.model.train(model_choice_index, model_reward)

    for generator in st.session_state.reward_generators:
        generator.potentially_change_mean()

    return (old_q, model_reward, new_q, model_choice_index)

st.set_page_config(layout="wide")
st.title("🎰 Non-Stationary 3-Armed Bandit Simulation")

if 'reward_generators' not in st.session_state:
    initialize_game()

left_column, right_column = st.columns(2)

(old_q, model_reward, new_q, model_choice_index) = (0, 0, 0, -1)


with left_column:

    st.header("Your Side")
    
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        if st.button("Pull Slot 1", use_container_width=True):
            (old_q, model_reward, new_q, model_choice_index) = play_turn(player_choice_index=0)
    with p_col2:
        if st.button("Pull Slot 2", use_container_width=True):
            (old_q, model_reward, new_q, model_choice_index) = play_turn(player_choice_index=1)
    with p_col3:
        if st.button("Pull Slot 3", use_container_width=True):
            (old_q, model_reward, new_q, model_choice_index) = play_turn(player_choice_index=2)

    st.metric(label="Your Last Reward", value=st.session_state.player_last_reward)

with right_column:

    st.header("Model's Side")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    cols = [m_col1, m_col2, m_col3]
    for i, col in enumerate(cols):
        with col:
            if st.session_state.model_last_choice == i:
                st.success(f"Slot {i+1} Pulled")
            else:
                st.info(f"Slot {i+1}")
    
    st.metric(label="Model's Last Reward", value=st.session_state.model_last_reward)

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reset Game", use_container_width=True):
        initialize_game()
        st.success("Game has been reset!")

    st.session_state.cheat_mode = st.toggle("🕵️ Cheat Mode", value=False)
    st.markdown("---")
    st.header("Game Scores")
    st.metric(label="Your Score", value=st.session_state.player.get_score())
    st.metric(label="Model's Score", value=st.session_state.model.get_score())
    st.markdown("---")
    st.info(
        "Each time you pull a lever, the model also pulls one. "
        "After each turn, the reward distributions may change."
    )

if st.session_state.cheat_mode:
    st.markdown("---")
    st.header("🕵️ Cheat Mode: Behind the Scenes")

    st.latex(r"Q_{n+1} = Q_n + \alpha(R_n - Q_n)")
    
    cheat_cols = st.columns(3)
    for i, col in enumerate(cheat_cols):
        with col:
            st.subheader(f"Slot {i+1} Details")
            true_mean = st.session_state.reward_generators[i].get_mean()
            st.write(f"**True Mean (E(X))**: `{true_mean:.2f}`")
            
            q_value = st.session_state.model.q_values[i]
            st.write(f"**Model's Estimate (Q)**: `{q_value:.2f}`")
    
    
    if model_choice_index != -1:

        with cheat_cols[model_choice_index]:

            st.write(f"*Q_n* (Old Estimate): `{old_q:.2f}`")
            st.write(f"*R_n* (Reward): `{model_reward}`")
            st.write(f"*α* (Learning Rate): `{st.session_state.model.learning_rate}`")
            st.write(f"New Q-value: {new_q:.2f}")