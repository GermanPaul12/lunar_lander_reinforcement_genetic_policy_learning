# lunar_lander_agents/test.py
import gymnasium as gym
import numpy as np
import torch
import random
import os
import argparse
import imageio # Import für GIF-Erstellung

import config
from agents import (
    AGENT_REGISTRY, DQNAgent, GeneticAgent, REINFORCEAgent,
    A2CAgent, PPOAgent, ESAgent
)
from agents.genetic_agent import PolicyNetwork as GAPolicyNetwork


GIF_DIR = os.path.join(config.PROJECT_ROOT, "gifs")
if not os.path.exists(GIF_DIR):
    try:
        os.makedirs(GIF_DIR)
        print(f"GIF-Verzeichnis erstellt: {GIF_DIR}")
    except OSError as e:
        print(f"Fehler beim Erstellen des GIF-Verzeichnisses {GIF_DIR}: {e}")


def setup_seeds():
    """Setzt Zufallsgeneratoren-Seeds für Reproduzierbarkeit."""
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    print(f"Seeds für Test (Agenteninitialisierung) gesetzt auf: {config.SEED}")


def run_test_episode_and_collect_frames(env, agent, current_seed):
    """Führt eine einzelne Testepisode durch und sammelt Frames für ein GIF."""
    frames_for_this_episode = [] # Frames nur für diese eine Episode
    observation, info = env.reset(seed=current_seed)
    agent.reset()

    total_reward = 0
    if env.render_mode == "rgb_array":
        frames_for_this_episode.append(env.render()) 

    for step in range(config.MAX_STEPS_PER_EVAL_EPISODE):
        if isinstance(agent, DQNAgent):
            action = agent.select_action(observation, eps=0.00)
        elif isinstance(agent, PPOAgent):
            action = agent.select_action(observation, store_in_memory=False)
        else:
            action = agent.select_action(observation)

        next_observation, reward, terminated, truncated, info = env.step(action)
        observation = next_observation
        total_reward += reward

        if env.render_mode == "rgb_array":
            frames_for_this_episode.append(env.render())
        elif env.render_mode == "human":
            env.render()

        if terminated or truncated:
            break
    return total_reward, step + 1, frames_for_this_episode


def main():
    parser = argparse.ArgumentParser(description="Test a pre-trained Lunar Lander agent and create a single GIF of multiple runs.")
    parser.add_argument(
        "--agent_type",
        type=str,
        required=True,
        choices=["random", "dqn", "genetic", "reinforce", "a2c", "ppo", "es"],
        help="Type of agent to test (e.g., dqn, ppo, random)."
    )
    parser.add_argument(
        "--num_runs_for_gif", # Umbenannt von num_gifs zu num_runs_for_gif
        type=int,
        default=config.NUM_TEST_RUNS, # Standardmäßig alle Testläufe für das GIF verwenden
        help=f"Number of test runs to include in the single GIF (max {config.NUM_TEST_RUNS} from config)."
    )
    parser.add_argument(
        "--gif_fps",
        type=int,
        default=30,
        help="Frames per second for the generated GIF."
    )
    args = parser.parse_args()

    config.ensure_model_dir_exists()
    setup_seeds()

    # Für GIF-Erstellung MUSS der render_mode "rgb_array" sein.
    # Wenn args.num_runs_for_gif > 0 ist, wird "rgb_array" verwendet.
    # Wenn args.num_runs_for_gif == 0, wird config.RENDER_MODE_TEST verwendet (z.B. "human" zum Zuschauen).
    render_mode_for_this_session = "rgb_array" if args.num_runs_for_gif > 0 else config.RENDER_MODE_TEST

    test_env = gym.make(config.ENV_ID, render_mode=render_mode_for_this_session)
    observation_space = test_env.observation_space
    action_space = test_env.action_space

    agent_to_test = None
    agent_name_lower = args.agent_type.lower()
    agent_name_upper = args.agent_type.upper()

    print(f"\n--- Preparing Agent for Testing: {agent_name_upper} ---")

    # Agentenlade-Logik (bleibt wie zuvor)
    if args.agent_type == "random":
        agent_to_test = AGENT_REGISTRY["random"](observation_space, action_space)
        print("Testing Random Agent.")
    elif args.agent_type == "dqn":
        if not os.path.exists(config.DQN_MODEL_PATH):
            print(f"ERROR: DQN model not found at {config.DQN_MODEL_PATH}. Train first.")
            test_env.close(); return
        agent_to_test = DQNAgent(observation_space, action_space, seed=config.SEED)
        try:
            agent_to_test.load(config.DQN_MODEL_PATH)
            agent_to_test.epsilon = 0.0
        except FileNotFoundError: test_env.close(); return
        print(f"Loaded DQN Agent from {config.DQN_MODEL_PATH}.")

    elif args.agent_type == "genetic":
        if not os.path.exists(config.GA_MODEL_PATH):
            print(f"ERROR: GA model not found at {config.GA_MODEL_PATH}. Train first.")
            test_env.close(); return
        best_policy_net = GAPolicyNetwork(
            state_size=observation_space.shape[0], action_size=action_space.n,
            seed=config.SEED, fc1_units=config.GA_FC1_UNITS, fc2_units=config.GA_FC2_UNITS
        ).to(config.DEVICE)
        try:
            best_policy_net.load_state_dict(torch.load(config.GA_MODEL_PATH, map_location=config.DEVICE))
            best_policy_net.eval()
            agent_to_test = GeneticAgent(observation_space, action_space, policy_network=best_policy_net, seed=config.SEED)
        except Exception as e: print(f"Error loading GA model: {e}"); test_env.close(); return
        print(f"Loaded Genetic Agent policy from {config.GA_MODEL_PATH}.")

    elif args.agent_type == "reinforce":
        if not os.path.exists(config.REINFORCE_MODEL_PATH):
            print(f"ERROR: REINFORCE model not found. Train first."); test_env.close(); return
        agent_to_test = REINFORCEAgent(
            observation_space, action_space, seed=config.SEED,
            fc1_units=config.REINFORCE_FC1_UNITS, fc2_units=config.REINFORCE_FC2_UNITS
        )
        try: agent_to_test.load(config.REINFORCE_MODEL_PATH)
        except FileNotFoundError: test_env.close(); return
        print(f"Loaded REINFORCE Agent from {config.REINFORCE_MODEL_PATH}.")

    elif args.agent_type == "a2c":
        if not os.path.exists(config.A2C_MODEL_PATH):
            print(f"ERROR: A2C model not found. Train first."); test_env.close(); return
        agent_to_test = A2CAgent(
            observation_space, action_space, seed=config.SEED,
            fc1_units=config.A2C_FC1_UNITS, fc2_units=config.A2C_FC2_UNITS
        )
        try: agent_to_test.load(config.A2C_MODEL_PATH)
        except FileNotFoundError: test_env.close(); return
        print(f"Loaded A2C Agent from {config.A2C_MODEL_PATH}.")

    elif args.agent_type == "ppo":
        if not (os.path.exists(config.PPO_ACTOR_MODEL_PATH) and os.path.exists(config.PPO_CRITIC_MODEL_PATH)):
            print(f"ERROR: PPO actor/critic model not found. Train first."); test_env.close(); return
        agent_to_test = PPOAgent(
            observation_space, action_space, seed=config.SEED,
            actor_fc1=config.PPO_ACTOR_FC1, actor_fc2=config.PPO_ACTOR_FC2,
            critic_fc1=config.PPO_CRITIC_FC1, critic_fc2=config.PPO_CRITIC_FC2
        )
        try: agent_to_test.load(config.PPO_ACTOR_MODEL_PATH, config.PPO_CRITIC_MODEL_PATH)
        except FileNotFoundError: test_env.close(); return
        print(f"Loaded PPO Agent from {config.PPO_ACTOR_MODEL_PATH} & {config.PPO_CRITIC_MODEL_PATH}.")

    elif args.agent_type == "es":
        if not os.path.exists(config.ES_MODEL_PATH):
            print(f"ERROR: ES model not found. Train first."); test_env.close(); return
        agent_to_test = ESAgent(
            observation_space, action_space, seed=config.SEED,
            fc1_units=config.ES_FC1_UNITS, fc2_units=config.ES_FC2_UNITS
        )
        try: agent_to_test.load(config.ES_MODEL_PATH)
        except FileNotFoundError: test_env.close(); return
        print(f"Loaded ES Agent (best policy) from {config.ES_MODEL_PATH}.")
            
    if agent_to_test is None:
        print(f"Could not load or initialize agent: {args.agent_type}")
        test_env.close(); return

    # Bestimme, wie viele Testläufe insgesamt gemacht werden und wie viele davon ins GIF kommen
    total_runs_to_perform = config.NUM_TEST_RUNS
    runs_for_gif_creation = min(args.num_runs_for_gif, total_runs_to_perform)

    print(f"\n--- Testing Agent: {agent_name_upper} for {total_runs_to_perform} runs ({runs_for_gif_creation} run(s) will be included in GIF) ---")
    
    all_test_rewards = []
    all_frames_for_single_gif = [] # Sammelt Frames über mehrere Episoden für EIN GIF

    for i_run in range(1, total_runs_to_perform + 1):
        current_run_seed = config.SEED + 1000 + i_run # Offset seeds für Varianz
        
        reward, steps, frames_from_episode = run_test_episode_and_collect_frames(
            test_env, agent_to_test, current_run_seed
        )
        all_test_rewards.append(reward)
        print(f"  Test Run {i_run}/{total_runs_to_perform}: Reward = {reward:.2f}, Steps = {steps}")

        # Frames sammeln, wenn dieser Lauf für das GIF vorgesehen ist und der Modus rgb_array ist
        if i_run <= runs_for_gif_creation and render_mode_for_this_session == "rgb_array" and frames_from_episode:
            all_frames_for_single_gif.extend(frames_from_episode)
            # Optional: Einen kurzen "Pause"-Frame oder einen Text-Frame zwischen den Episoden einfügen
            # if i_run < runs_for_gif_creation and frames_from_episode:
            #     # Erstelle einen schwarzen Frame oder einen Frame mit Text "Episode X beendet"
            #     # Dies erfordert zusätzliche Logik mit z.B. Pillow oder OpenCV, um Text auf Bilder zu zeichnen.
            #     # Für Einfachheit hier weggelassen.
            #     # Beispiel: all_frames_for_single_gif.extend([frames_from_episode[-1]] * 10) # Letzten Frame 10x wiederholen
            #     pass


    # GIF speichern, NACHDEM alle vorgesehenen Läufe abgeschlossen sind und Frames gesammelt wurden
    if all_frames_for_single_gif and render_mode_for_this_session == "rgb_array":
        # Dateiname für das kombinierte GIF
        # Es ist besser, den Seed des ersten Laufs im GIF oder einen Zeitstempel zu verwenden,
        # da current_run_seed sich ändert. Oder einfach nur den Agententyp.
        first_gif_run_seed = config.SEED + 1000 + 1 
        gif_filename = os.path.join(GIF_DIR, f"{agent_name_lower}_combined_{runs_for_gif_creation}runs_seed{first_gif_run_seed}.gif")
        try:
            imageio.mimsave(gif_filename, all_frames_for_single_gif, fps=args.gif_fps)
            print(f"    Kombiniertes GIF gespeichert: {gif_filename}")
        except Exception as e:
            print(f"    Fehler beim Speichern des kombinierten GIFs {gif_filename}: {e}")
    
    test_env.close()
    
    if all_test_rewards:
        print("\n--- Test Summary ---")
        print(f"Agent: {agent_name_upper}")
        print(f"Average Reward over {total_runs_to_perform} runs: {np.mean(all_test_rewards):.2f}")
        print(f"Min Reward: {np.min(all_test_rewards):.2f}, Max Reward: {np.max(all_test_rewards):.2f}")

if __name__ == "__main__":
    main()