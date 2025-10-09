from Algorithm import AlgorithmBaseClass
import numpy as np

class QLearningAlgorithm(AlgorithmBaseClass):
    def __init__(self, cfg=None):
        super().__init__()
        self.horizontal_actions = ["TURN_RIGHT_FAST", "TURN_RIGHT_SLOW", "TURN_LEFT_FAST", "TURN_LEFT_SLOW"]
        self.vertical_actions = ["CLIMB", "DESCEND"]
        self.actions = [(h, v) for h in self.horizontal_actions for v in self.vertical_actions]
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.0

    def compute_risk_profile(self, *args, **kwargs):
        return None

    def state_to_tuple(self, range, sector_prob, yaw, pitch, roll, altitude, longitude, latitude):
        return (
            round(range, 1),
            round(sector_prob, 2),
            round(yaw, 1),
            round(pitch, 1),
            round(roll, 1),
            round(altitude, 1),
            round(longitude, 6),   # 经纬度保留6位小数
            round(latitude, 6)
        )

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.actions[np.random.randint(len(self.actions))]
        qs = [self.q_table.get((state, a), 0) for a in self.actions]
        return self.actions[np.argmax(qs)]

    def update(self, state, action, reward, next_state):
        q_predict = self.q_table.get((state, action), 0)
        q_target = reward + self.gamma * max([self.q_table.get((next_state, a), 0) for a in self.actions])
        self.q_table[(state, action)] = q_predict + self.alpha * (q_target - q_predict)

def compute_reward(
    prev_distance, curr_distance, curr_sector_prob,
    prev_yaw, curr_yaw, prev_pitch, curr_pitch, prev_roll, curr_roll,
    prev_altitude, curr_altitude,
    prev_longitude, curr_longitude, prev_latitude, curr_latitude,
    action,
    w1=1.0, w2=2.0, w3=0.2, w4=0.2, w5=0.5, w6=0.1, right_fast_bonus=2.0
):
    # 距离减少为正，碰撞概率为负，姿态变化为负，高度变化为正，经纬度变化可选
    reward = (
        w1 * (prev_distance - curr_distance)
        - w2 * curr_sector_prob
        - w3 * abs(curr_yaw - prev_yaw) / 180
        - w4 * abs(curr_pitch - prev_pitch) / 90
        - w4 * abs(curr_roll - prev_roll) / 90
        + w5 * (curr_altitude - prev_altitude)
        - w6 * (abs(curr_longitude - prev_longitude) + abs(curr_latitude - prev_latitude))  # 可选项
    )
    if action[0] == "TURN_RIGHT_FAST":
        reward += right_fast_bonus
    # 爬升比下降有更多奖励
    if action[1] == "CLIMB":
        reward += 1.0  # 可根据需要调整奖励值
    elif action[1] == "DESCEND":
        reward -= 0.5  # 可根据需要调整惩罚值
    return reward

def train_qlearning(daaResult, uwb_data, sector_probs, yaws, pitchs, rolls, altitudes, longitudes, latitudes):
    algo = QLearningAlgorithm()
    actions = []
    for i in range(1, len(uwb_data)):
        prev = uwb_data[i-1]
        curr = uwb_data[i]
        state = algo.state_to_tuple(
            prev["range"],
            sector_probs[i-1],
            yaws[i-1],
            pitchs[i-1],
            rolls[i-1],
            altitudes[i-1],
            longitudes[i-1],
            latitudes[i-1]
        )
        next_state = algo.state_to_tuple(
            curr["range"],
            sector_probs[i],
            yaws[i],
            pitchs[i],
            rolls[i],
            altitudes[i],
            longitudes[i],
            latitudes[i]
        )
        for action in algo.actions:
            reward = compute_reward(
                prev["range"], curr["range"],
                sector_probs[i],
                yaws[i-1], yaws[i],
                pitchs[i-1], pitchs[i],
                rolls[i-1], rolls[i],
                altitudes[i-1], altitudes[i],
                longitudes[i-1], longitudes[i],
                latitudes[i-1], latitudes[i],
                action
            )
            algo.update(state, action, reward, next_state)
        chosen_action = algo.select_action(state)
        actions.append(chosen_action)
    # 最后一个状态补一个动作
    last = uwb_data[-1]
    last_state = algo.state_to_tuple(
        last["range"], sector_probs[-1], yaws[-1], pitchs[-1], rolls[-1], altitudes[-1], longitudes[-1], latitudes[-1]
    )
    actions.append(algo.select_action(last_state))
    return algo, actions
