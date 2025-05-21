# lunar_lander_agents/train.py
import gymnasium as gym
import numpy as np
import torch
import random
import os
from collections import deque

import config # Globale Konfigurationen importieren
# Alle Agentenklassen und zugehörige Controller/Netzwerke importieren
from agents import (
    AGENT_REGISTRY, DQNAgent, GeneticAlgorithmController, GeneticAgent,
    REINFORCEAgent, A2CAgent, PPOAgent, ESAgent
)

def setup_seeds():
    """Setzt Zufallsgeneratoren-Seeds für Reproduzierbarkeit."""
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available(): # Falls CUDA (GPU) verfügbar ist
        torch.cuda.manual_seed_all(config.SEED) # Seed auch für alle GPUs setzen
        # Die folgenden Zeilen können für striktere Reproduzierbarkeit auf Kosten der Performance sorgen
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Seeds gesetzt auf: {config.SEED}")

def train_dqn_agent(observation_space, action_space):
    """Trainiert den DQN-Agenten."""
    print("\n--- Training DQN Agent ---")
    # DQN-Agent mit Parametern aus config.py initialisieren
    dqn_agent = DQNAgent(observation_space, action_space, seed=config.SEED)

    scores_deque = deque(maxlen=100) # Deque zur Speicherung der letzten 100 Scores für gleitenden Durchschnitt
    # scores_all_episodes = [] # Optional: Alle Scores für spätere Plots speichern

    train_env = gym.make(config.ENV_ID) # Separate Trainingsumgebung erstellen
    print(f"Starte DQN Training für {config.DQN_TRAIN_EPISODES} Episoden...")

    for i_episode in range(1, config.DQN_TRAIN_EPISODES + 1):
        obs, _ = train_env.reset(seed=config.SEED + i_episode) # Umgebung zurücksetzen, Seed variieren
        score = 0
        dqn_agent.reset() # Epsilon-Wert des Agenten anpassen (Decay)

        for t in range(config.DQN_MAX_T_PER_EPISODE): # Schleife über maximale Schritte pro Episode
            action = dqn_agent.select_action(obs) # Aktion basierend auf aktueller Policy wählen
            next_obs, reward, terminated, truncated, _ = train_env.step(action) # Aktion ausführen
            
            # Agent lernt aus der gemachten Erfahrung
            dqn_agent.learn(obs, action, reward, next_obs, terminated, truncated)
            
            obs = next_obs # Zustand aktualisieren
            score += reward # Belohnung akkumulieren
            if terminated or truncated: # Episode beendet?
                break
        
        scores_deque.append(score)
        # scores_all_episodes.append(score)

        # Fortschritt periodisch ausgeben
        if i_episode % config.DQN_PRINT_EVERY == 0 or i_episode == config.DQN_TRAIN_EPISODES:
            avg_score = np.mean(scores_deque) if scores_deque else score
            print(f'\rEpisode {i_episode}/{config.DQN_TRAIN_EPISODES}\tAvg Score (letzte 100): {avg_score:.2f}\tEpsilon: {dqn_agent.epsilon:.4f}')
        
        # Prüfen, ob das Ziel erreicht wurde (stabiler Durchschnittsscore)
        if len(scores_deque) >= 100 and np.mean(scores_deque) >= config.DQN_SCORE_TARGET:
            print(f'\nUmgebung in {i_episode-100:d} Episoden gelöst! Avg Score: {np.mean(scores_deque):.2f}')
            break # Training vorzeitig beenden
            
    train_env.close() # Trainingsumgebung schließen
    print(f"DQN Training beendet. Speichere Modell...")
    dqn_agent.save(config.DQN_MODEL_PATH) # Trainiertes Modell speichern
    return dqn_agent

def train_ga_agent(observation_space, action_space):
    """Trainiert den Genetischen Algorithmus Agenten."""
    print("\n--- Training Genetischer Algorithmus Agent ---")
    # GA-Controller mit Parametern aus config.py initialisieren
    ga_controller = GeneticAlgorithmController(
        state_size=observation_space.shape[0],
        action_size=action_space.n,
        env_id=config.ENV_ID,
        n_generations=config.GA_N_GENERATIONS,
        population_size=config.GA_POPULATION_SIZE,
        eval_episodes_per_individual=config.GA_EVAL_EPISODES_PER_INDIVIDUAL,
        max_steps_per_eval_episode=config.GA_MAX_STEPS_PER_GA_EVAL,
        mutation_rate=config.GA_MUTATION_RATE,
        mutation_strength=config.GA_MUTATION_STRENGTH,
        crossover_rate=config.GA_CROSSOVER_RATE,
        tournament_size=config.GA_TOURNAMENT_SIZE,
        elitism_count=config.GA_ELITISM_COUNT,
        seed=config.SEED
    )

    print(f"Starte GA Training für {config.GA_N_GENERATIONS} Generationen...")
    for gen in range(config.GA_N_GENERATIONS): # Schleife über Anzahl der Generationen
        mean_fitness, max_fitness = ga_controller.evolve_population() # Eine Generation evolvieren
        print(f"Generation {gen + 1}/{config.GA_N_GENERATIONS} - Mean Fitness: {mean_fitness:.2f}, Max Fitness: {max_fitness:.2f}, Best Overall: {ga_controller.best_fitness:.2f}")
        
        # Bestes Individuum periodisch oder am Ende speichern
        if (gen + 1) % config.GA_SAVE_EVERY_N_GENERATIONS == 0 or (gen + 1) == config.GA_N_GENERATIONS:
            ga_controller.save_best_individual(config.GA_MODEL_PATH)

    best_policy_net = ga_controller.get_best_policy_network() # Beste gefundene Policy abrufen
    if best_policy_net:
        print("GA Training beendet. Beste Policy gefunden.")
        # GeneticAgent-Wrapper mit der besten Policy erstellen
        return GeneticAgent(observation_space, action_space, policy_network=best_policy_net, seed=config.SEED)
    else:
        print("GA Training beendet, aber keine verwendbare Policy gefunden.")
        return None

def train_reinforce_agent(observation_space, action_space):
    """Trainiert den REINFORCE Agenten."""
    print("\n--- Training REINFORCE Agent ---")
    reinforce_agent = REINFORCEAgent(
        observation_space, action_space, seed=config.SEED,
        learning_rate=config.REINFORCE_LEARNING_RATE,
        gamma=config.REINFORCE_GAMMA,
        fc1_units=config.REINFORCE_FC1_UNITS,
        fc2_units=config.REINFORCE_FC2_UNITS
    )

    scores_deque = deque(maxlen=100)
    train_env = gym.make(config.ENV_ID)
    print(f"Starte REINFORCE Training für {config.REINFORCE_TRAIN_EPISODES} Episoden...")

    for i_episode in range(1, config.REINFORCE_TRAIN_EPISODES + 1):
        obs, _ = train_env.reset(seed=config.SEED + i_episode)
        reinforce_agent.reset() # Löscht gesammelte Log-Probs und Rewards der letzten Episode
        episode_score = 0

        for t in range(config.REINFORCE_MAX_T_PER_EPISODE):
            action = reinforce_agent.select_action(obs) # Aktion wählen, speichert log_prob intern
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            
            reinforce_agent.store_reward(reward) # Belohnung für diesen Schritt speichern
            
            obs = next_obs
            episode_score += reward
            if terminated or truncated:
                break
        
        # Lernupdate am Ende der Episode durchführen
        reinforce_agent.learn_episode() 
        
        scores_deque.append(episode_score)

        if i_episode % config.REINFORCE_PRINT_EVERY == 0 or i_episode == config.REINFORCE_TRAIN_EPISODES:
            avg_score = np.mean(scores_deque) if scores_deque else episode_score
            print(f'\rEpisode {i_episode}/{config.REINFORCE_TRAIN_EPISODES}\tAvg Score (letzte 100): {avg_score:.2f}')
        
        if len(scores_deque) >= 100 and np.mean(scores_deque) >= config.REINFORCE_SCORE_TARGET:
            print(f'\nUmgebung mit REINFORCE (Ziel {config.REINFORCE_SCORE_TARGET}) in {i_episode-100:d} Episoden gelöst! Avg Score: {np.mean(scores_deque):.2f}')
            break
            
    train_env.close()
    print(f"REINFORCE Training beendet. Speichere Modell...")
    reinforce_agent.save(config.REINFORCE_MODEL_PATH)
    return reinforce_agent

def train_a2c_agent(observation_space, action_space):
    """Trainiert den A2C Agenten."""
    print("\n--- Training A2C Agent ---")
    a2c_agent = A2CAgent(
        observation_space, action_space, seed=config.SEED,
        learning_rate=config.A2C_LEARNING_RATE,
        gamma=config.A2C_GAMMA,
        entropy_coeff=config.A2C_ENTROPY_COEFF, # Koeffizient für den Entropie-Bonus
        value_loss_coeff=config.A2C_VALUE_LOSS_COEFF, # Gewichtung des Value-Loss
        fc1_units=config.A2C_FC1_UNITS,
        fc2_units=config.A2C_FC2_UNITS
    )

    scores_deque = deque(maxlen=100)
    train_env = gym.make(config.ENV_ID)
    print(f"Starte A2C Training für {config.A2C_TRAIN_EPISODES} Episoden...")

    for i_episode in range(1, config.A2C_TRAIN_EPISODES + 1):
        obs, _ = train_env.reset(seed=config.SEED + i_episode)
        a2c_agent.reset() # Für A2C typischerweise keine spezielle Reset-Logik nötig
        episode_score = 0

        for t in range(config.A2C_MAX_T_PER_EPISODE):
            action = a2c_agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            
            # A2C lernt nach jedem Schritt (on-policy)
            a2c_agent.learn(obs, action, reward, next_obs, terminated, truncated)
            
            obs = next_obs
            episode_score += reward
            if terminated or truncated:
                break
        
        scores_deque.append(episode_score)

        if i_episode % config.A2C_PRINT_EVERY == 0 or i_episode == config.A2C_TRAIN_EPISODES:
            avg_score = np.mean(scores_deque) if scores_deque else episode_score
            print(f'\rEpisode {i_episode}/{config.A2C_TRAIN_EPISODES}\tAvg Score (letzte 100): {avg_score:.2f}')
        
        if len(scores_deque) >= 100 and np.mean(scores_deque) >= config.A2C_SCORE_TARGET:
            print(f'\nUmgebung mit A2C (Ziel {config.A2C_SCORE_TARGET}) in {i_episode-100:d} Episoden gelöst! Avg Score: {np.mean(scores_deque):.2f}')
            break
            
    train_env.close()
    print(f"A2C Training beendet. Speichere Modell...")
    a2c_agent.save(config.A2C_MODEL_PATH)
    return a2c_agent

def train_ppo_agent(observation_space, action_space):
    """Trainiert den PPO Agenten."""
    print("\n--- Training PPO Agent ---")
    ppo_agent = PPOAgent(
        observation_space, action_space, seed=config.SEED,
        actor_lr=config.PPO_ACTOR_LR, critic_lr=config.PPO_CRITIC_LR,
        gamma=config.PPO_GAMMA, ppo_epochs=config.PPO_EPOCHS,
        ppo_clip=config.PPO_CLIP_EPSILON, batch_size=config.PPO_BATCH_SIZE,
        gae_lambda=config.PPO_GAE_LAMBDA, entropy_coeff=config.PPO_ENTROPY_COEFF,
        value_loss_coeff=config.PPO_VALUE_LOSS_COEFF,
        actor_fc1=config.PPO_ACTOR_FC1, actor_fc2=config.PPO_ACTOR_FC2,
        critic_fc1=config.PPO_CRITIC_FC1, critic_fc2=config.PPO_CRITIC_FC2,
        update_horizon=config.PPO_UPDATE_HORIZON # Anzahl Schritte pro Datensammlung vor Update
    )

    train_env = gym.make(config.ENV_ID)
    print(f"Starte PPO Training für ca. {config.PPO_TOTAL_TIMESTEPS} Zeitschritte...")
    
    obs, _ = train_env.reset(seed=config.SEED)
    current_episode_reward = 0
    # current_episode_length = 0 # Nicht unbedingt für Logik benötigt, aber für Debugging
    completed_episodes = 0
    scores_deque = deque(maxlen=100) # Für Durchschnittsbelohnung der letzten 100 Episoden
    updates_done = 0

    for timestep in range(1, config.PPO_TOTAL_TIMESTEPS + 1):
        # Aktion wählen und relevante Daten im Agenten-Speicher ablegen
        action = ppo_agent.select_action(obs, store_in_memory=True) 
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        
        # Ergebnis des Schritts (Belohnung, Beendigung) im Agenten-Speicher ablegen
        ppo_agent.store_transition_result(reward, terminated or truncated) 
        
        obs = next_obs
        current_episode_reward += reward
        # current_episode_length +=1

        if terminated or truncated: # Episode beendet
            scores_deque.append(current_episode_reward)
            completed_episodes += 1
            obs, _ = train_env.reset(seed=config.SEED + completed_episodes) # Umgebung zurücksetzen
            current_episode_reward = 0
            # current_episode_length = 0

        # Prüfen, ob genügend Daten für ein PPO-Update gesammelt wurden
        if len(ppo_agent.memory_actions) >= config.PPO_UPDATE_HORIZON:
            # Letzte Beobachtung übergeben, falls Horizont mitten in Episode endet (für V(s_T) Schätzung)
            last_obs_for_value_est = obs if not (terminated or truncated) else None
            ppo_agent.learn_from_memory(last_obs_for_value_est) # PPO Lernschritt durchführen
            updates_done += 1

            # Fortschritt periodisch ausgeben und Modell speichern
            if updates_done % config.PPO_PRINT_EVERY_N_UPDATES == 0 and scores_deque:
                avg_score = np.mean(scores_deque)
                print(f"Zeitschritt: {timestep}/{config.PPO_TOTAL_TIMESTEPS} | Updates: {updates_done} | Avg Reward (letzte 100 Eps): {avg_score:.2f}")
                ppo_agent.save(config.PPO_ACTOR_MODEL_PATH, config.PPO_CRITIC_MODEL_PATH)

            # Ziel erreicht?
            if scores_deque and np.mean(scores_deque) >= config.PPO_SCORE_TARGET and len(scores_deque) >= 100:
                print(f"\nUmgebung mit PPO (Ziel {config.PPO_SCORE_TARGET}) gelöst! Avg Score: {np.mean(scores_deque):.2f}")
                break # Training beenden
    
    train_env.close()
    print(f"PPO Training beendet. Speichere finales Modell...")
    ppo_agent.save(config.PPO_ACTOR_MODEL_PATH, config.PPO_CRITIC_MODEL_PATH)
    return ppo_agent

def train_es_agent(observation_space, action_space):
    """Trainiert den Evolutionäre Strategien (ES) Agenten."""
    print("\n--- Training Evolutionäre Strategien (ES) Agent ---")
    # Der ESAgent ist hier der Controller, der die zentrale Policy optimiert
    es_agent_controller = ESAgent( 
        observation_space, action_space, seed=config.SEED,
        population_size=config.ES_POPULATION_SIZE, # Anzahl der Perturbationen
        sigma=config.ES_SIGMA,                     # Stärke der Perturbationen
        learning_rate=config.ES_LEARNING_RATE,     # Lernrate für Update der zentralen Gewichte
        eval_episodes_per_param=config.ES_EVAL_EPISODES_PER_PARAM, # Episoden zur Fitness-Evaluierung
        fc1_units=config.ES_FC1_UNITS,
        fc2_units=config.ES_FC2_UNITS
    )

    # ES benötigt eine Umgebung für seine internen Fitness-Evaluationen
    eval_env_for_es = gym.make(config.ENV_ID) 
    
    print(f"Starte ES Training für {config.ES_N_GENERATIONS} Generationen...")
    # avg_fitness_history = [] # Optional: Verlauf der Fitness speichern

    for gen in range(1, config.ES_N_GENERATIONS + 1):
        # Ein Evolutionsschritt: Perturbationen erzeugen, evaluieren, zentrale Gewichte anpassen
        mean_pop_fitness, max_pop_fitness, central_fitness = es_agent_controller.evolve_step(eval_env_for_es)
        # avg_fitness_history.append(central_fitness)

        # Fortschritt ausgeben und periodisch speichern
        if gen % config.ES_PRINT_EVERY == 0 or gen == config.ES_N_GENERATIONS:
            print(f"ES Gen {gen}/{config.ES_N_GENERATIONS} | Mean Pop Fit: {mean_pop_fitness:.2f} | Max Pop Fit: {max_pop_fitness:.2f} | Central Fit: {central_fitness:.2f} | Best Overall: {es_agent_controller.current_best_fitness:.2f}")
            es_agent_controller.save(config.ES_MODEL_PATH) # Beste bisher gefundene Policy speichern

        # Ziel erreicht?
        if es_agent_controller.current_best_fitness >= config.ES_SCORE_TARGET:
            print(f"\nES Ziel-Score erreicht! Beste Fitness: {es_agent_controller.current_best_fitness:.2f}")
            break
            
    eval_env_for_es.close()
    print(f"ES Training beendet. Speichere finales bestes Modell...")
    es_agent_controller.save(config.ES_MODEL_PATH)
    return es_agent_controller # Der ESAgent (Controller) hält die beste Policy

def main():
    """Hauptfunktion: Initialisiert Seeds, Umgebungsinformationen und startet Trainingsprozesse."""
    config.ensure_model_dir_exists() # Stellt sicher, dass das model/ Verzeichnis existiert
    setup_seeds() # Setzt alle Zufallsgeneratoren-Seeds

    # Temporäre Umgebung, um Beobachtungs- und Aktionsraum-Dimensionen zu erhalten
    temp_env = gym.make(config.ENV_ID)
    observation_space = temp_env.observation_space
    action_space = temp_env.action_space
    temp_env.close()

    # Iteriert über alle in config.py definierten Agententypen
    for agent_type in config.AGENT_TYPES:
        if agent_type == "random": # Zufallsagent benötigt kein Training
            print("\n--- Verarbeitung Agent: RANDOM ---")
            print("Zufallsagent benötigt kein Training.")
            continue

        print(f"\n--- Verarbeitung Agent: {agent_type.upper()} ---")
        
        force_retrain = False # Standardmäßig nicht neu trainieren
        model_path = None     # Pfad zum (Haupt-)Modell des Agenten

        # Spezifische Konfigurationen für jeden Agententyp laden
        if agent_type == "dqn":
            force_retrain = config.FORCE_RETRAIN_DQN
            model_path = config.DQN_MODEL_PATH
        elif agent_type == "genetic":
            force_retrain = config.FORCE_RETRAIN_GA
            model_path = config.GA_MODEL_PATH
        elif agent_type == "reinforce": 
            force_retrain = config.FORCE_RETRAIN_REINFORCE
            model_path = config.REINFORCE_MODEL_PATH
        elif agent_type == "a2c": 
            force_retrain = config.FORCE_RETRAIN_A2C
            model_path = config.A2C_MODEL_PATH
        elif agent_type == "ppo": 
            force_retrain = config.FORCE_RETRAIN_PPO
            # PPO hat zwei Modelle; Existenz des Actor-Modells als Indikator
            model_path = config.PPO_ACTOR_MODEL_PATH 
        elif agent_type == "es":
            force_retrain = config.FORCE_RETRAIN_ES
            model_path = config.ES_MODEL_PATH
        
        # Überspringen, falls kein Modellpfad definiert (sollte nicht passieren für lernfähige Agenten)
        if not model_path and agent_type not in ["random"]:
            print(f"Warnung: Kein Modellpfad für Agententyp {agent_type} definiert. Überspringe.")
            continue

        model_exists = os.path.exists(model_path) if model_path else False

        # Entscheidung: Trainieren oder Laden?
        if not force_retrain and model_exists:
            print(f"Vortrainiertes Modell für {agent_type.upper()} unter {model_path} gefunden. Training übersprungen.")
        else:
            if force_retrain and model_exists:
                print(f"Erzwinge Neutraining für {agent_type.upper()}, obwohl Modell unter {model_path} existiert.")
            elif not model_exists:
                print(f"Kein vortrainiertes Modell für {agent_type.upper()} unter {model_path} gefunden. Starte Training.")
            
            # Aufruf der spezifischen Trainingsfunktion
            if agent_type == "dqn":
                train_dqn_agent(observation_space, action_space)
            elif agent_type == "genetic":
                train_ga_agent(observation_space, action_space)
            elif agent_type == "reinforce": 
                train_reinforce_agent(observation_space, action_space)
            elif agent_type == "a2c": 
                train_a2c_agent(observation_space, action_space)
            elif agent_type == "ppo": 
                train_ppo_agent(observation_space, action_space)
            elif agent_type == "es": 
                train_es_agent(observation_space, action_space)
    
    print("\nAlle spezifizierten Trainingsprozesse abgeschlossen.")

if __name__ == "__main__":
    main()