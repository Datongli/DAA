import numpy as np
from QLearningAlgorithm import (
    QLearningAgent, UAVCollisionEnv, evaluate, animate_episode, QTABLE_PATH
)

def main():
    agent = QLearningAgent()
    # 加载训练好的 Q 表
    if QTABLE_PATH.exists():
        agent.Q = np.load(QTABLE_PATH)
        print(f"[QTable] loaded from: {QTABLE_PATH}")
    else:
        print(f"[QTable] not found: {QTABLE_PATH}")
        return

    env = UAVCollisionEnv(seed=0)
    # 使用贪婪策略跑一次仿真并动态演示
    result = evaluate(agent, env, render=False)
    print("Final info:", result["final_info"])
    animate_episode(result["own_traj"], result["intr_traj"], interval_ms=100, save_path=None)

if __name__ == "__main__":
    main()