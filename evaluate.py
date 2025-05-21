# lunar_lander_agents/evaluate.py
import gymnasium as gym
import numpy as np
import torch
import random
import os
import csv # Für das Schreiben der CSV-Datei

import config # Globale Konfigurationen importieren

# Alle Agentenklassen und ggf. spezifische Policy-Netzwerke importieren
from agents import (
    AGENT_REGISTRY, DQNAgent, GeneticAgent, REINFORCEAgent,
    A2CAgent, PPOAgent, ESAgent
    # PolicyNetwork wird hier nicht mehr direkt importiert, da Ladelogik spezifisch ist
)
# Importiere PolicyNetwork spezifisch für GA, um Klarheit zu schaffen, falls andere Agenten
# auch eine Klasse namens PolicyNetwork haben (obwohl sie hier anders benannt wurden).
from agents.genetic_agent import PolicyNetwork as GAPolicyNetwork

def setup_seeds():
    """Setzt Zufallsgeneratoren-Seeds für Reproduzierbarkeit der Evaluierungsläufe."""
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available(): # Falls CUDA (GPU) verfügbar ist
        torch.cuda.manual_seed_all(config.SEED) # Seed auch für alle GPUs setzen
    print(f"Seeds für Evaluierung gesetzt auf: {config.SEED}")

def run_evaluation_episode(env, agent, current_seed):
    """Führt eine einzelne Evaluierungsepisode für einen gegebenen Agenten durch."""
    observation, info = env.reset(seed=current_seed) # Umgebung mit spezifischem Seed zurücksetzen
    agent.reset() # Internen Zustand des Agenten zurücksetzen (z.B. Epsilon für DQN)

    total_reward = 0
    # Schleife über maximale Schritte pro Episode (aus config.py)
    for step in range(config.MAX_STEPS_PER_EVAL_EPISODE):
        if isinstance(agent, DQNAgent):
            # Für DQN im Evaluierungsmodus: geringe Exploration (epsilon = 0.01)
            action = agent.select_action(observation, eps=0.01) 
        elif isinstance(agent, PPOAgent):
            # PPO's select_action hat einen Parameter store_in_memory, der beim Evaluieren False sein sollte
            action = agent.select_action(observation, store_in_memory=False)
        else:
            action = agent.select_action(observation) # Standard-Aktionsauswahl für andere Agenten

        next_observation, reward, terminated, truncated, _ = env.step(action) # Aktion ausführen
        observation = next_observation # Zustand aktualisieren
        total_reward += reward # Belohnung akkumulieren

        if env.render_mode == "human": # Rendern, falls im config.py für Evaluierung aktiviert
            env.render()
        
        if terminated or truncated: # Episode beendet?
            break
    return total_reward, step + 1 # Gesamte Belohnung und Anzahl der Schritte zurückgeben

def load_agent(agent_type, observation_space, action_space):
    """Lädt einen spezifischen, vortrainierten Agenten für die Evaluierung.
    
    Args:
        agent_type (str): Der Typ des zu ladenden Agenten (z.B. "dqn", "ppo").
        observation_space: Der Beobachtungsraum der Umgebung.
        action_space: Der Aktionsraum der Umgebung.

    Returns:
        Eine Instanz des geladenen Agenten oder None, falls das Laden fehlschlägt.
    """
    agent_instance = None
    print(f"Versuche, Agenten zu laden: {agent_type.upper()}")

    if agent_type == "random":
        agent_instance = AGENT_REGISTRY["random"](observation_space, action_space)
        print("Zufallsagent initialisiert.")
    elif agent_type == "dqn":
        if not os.path.exists(config.DQN_MODEL_PATH): # Prüfen, ob Modelldatei existiert
            print(f"WARNUNG: DQN-Modell nicht unter {config.DQN_MODEL_PATH} gefunden. Überspringe DQN."); return None
        # DQN-Agent mit Parametern aus config.py (hauptsächlich Seed und Netzwerkarchitektur) initialisieren
        agent_instance = DQNAgent(observation_space, action_space, seed=config.SEED)
        try:
            agent_instance.load(config.DQN_MODEL_PATH) # Modell laden
            agent_instance.epsilon = 0.01 # Für Evaluierung konsistentes, niedriges Epsilon
        except FileNotFoundError: return None # Falls Laden fehlschlägt
        print(f"DQN Agent geladen von {config.DQN_MODEL_PATH}.")
    elif agent_type == "genetic":
        if not os.path.exists(config.GA_MODEL_PATH):
            print(f"WARNUNG: GA-Modell nicht unter {config.GA_MODEL_PATH} gefunden. Überspringe GA."); return None
        # GA-Policy-Netzwerk mit korrekter Architektur aus config.py instanziieren
        best_policy_net = GAPolicyNetwork( # Spezifischer Import für Klarheit
            state_size=observation_space.shape[0], action_size=action_space.n,
            seed=config.SEED, 
            fc1_units=config.GA_FC1_UNITS, # Wichtig: Muss mit trainiertem Modell übereinstimmen
            fc2_units=config.GA_FC2_UNITS  # Wichtig: Muss mit trainiertem Modell übereinstimmen
        ).to(config.DEVICE)
        try:
            best_policy_net.load_state_dict(torch.load(config.GA_MODEL_PATH, map_location=config.DEVICE))
            best_policy_net.eval() # Netzwerk in Evaluationsmodus setzen
            # GeneticAgent-Wrapper mit der geladenen besten Policy erstellen
            agent_instance = GeneticAgent(observation_space, action_space, policy_network=best_policy_net, seed=config.SEED)
        except Exception as e: print(f"Fehler beim Laden des GA-Modells: {e}. Überspringe GA."); return None
        print(f"Genetic Agent Policy geladen von {config.GA_MODEL_PATH}.")
    elif agent_type == "reinforce":
        if not os.path.exists(config.REINFORCE_MODEL_PATH):
            print(f"WARNUNG: REINFORCE-Modell nicht gefunden. Überspringe REINFORCE."); return None
        # REINFORCEAgent mit korrekter Netzwerkarchitektur instanziieren
        agent_instance = REINFORCEAgent(
            observation_space, action_space, seed=config.SEED,
            learning_rate=config.REINFORCE_LEARNING_RATE, # LR/Gamma nicht für reine Aktionsauswahl relevant
            gamma=config.REINFORCE_GAMMA,
            fc1_units=config.REINFORCE_FC1_UNITS, # Wichtig für korrekte Modellstruktur
            fc2_units=config.REINFORCE_FC2_UNITS
        )
        try: agent_instance.load(config.REINFORCE_MODEL_PATH)
        except FileNotFoundError: return None
        print(f"REINFORCE Agent geladen von {config.REINFORCE_MODEL_PATH}.")
    elif agent_type == "a2c":
        if not os.path.exists(config.A2C_MODEL_PATH):
            print(f"WARNUNG: A2C-Modell nicht gefunden. Überspringe A2C."); return None
        # A2CAgent mit korrekter Netzwerkarchitektur instanziieren
        agent_instance = A2CAgent(
            observation_space, action_space, seed=config.SEED,
            # Trainingshyperparameter hier weniger relevant, aber Netzwerkstruktur muss stimmen
            fc1_units=config.A2C_FC1_UNITS, 
            fc2_units=config.A2C_FC2_UNITS
        )
        try: agent_instance.load(config.A2C_MODEL_PATH)
        except FileNotFoundError: return None
        print(f"A2C Agent geladen von {config.A2C_MODEL_PATH}.")
    elif agent_type == "ppo":
        # PPO benötigt separate Actor- und Critic-Modelle
        if not (os.path.exists(config.PPO_ACTOR_MODEL_PATH) and os.path.exists(config.PPO_CRITIC_MODEL_PATH)):
            print(f"WARNUNG: PPO Actor/Critic-Modell nicht gefunden. Überspringe PPO."); return None
        # PPOAgent mit korrekter Netzwerkarchitektur instanziieren
        agent_instance = PPOAgent(
            observation_space, action_space, seed=config.SEED,
            # Viele PPO-Parameter sind nur für das Training, aber Netzwerkarchitektur ist wichtig
            actor_fc1=config.PPO_ACTOR_FC1, actor_fc2=config.PPO_ACTOR_FC2,
            critic_fc1=config.PPO_CRITIC_FC1, critic_fc2=config.PPO_CRITIC_FC2
        )
        try: agent_instance.load(config.PPO_ACTOR_MODEL_PATH, config.PPO_CRITIC_MODEL_PATH)
        except FileNotFoundError: return None
        print(f"PPO Agent geladen von {config.PPO_ACTOR_MODEL_PATH} & {config.PPO_CRITIC_MODEL_PATH}.")
    elif agent_type == "es":
        if not os.path.exists(config.ES_MODEL_PATH):
            print(f"WARNUNG: ES-Modell nicht gefunden. Überspringe ES."); return None
        # Der ESAgent (Controller) hält die beste Policy und agiert als Agent für die Evaluierung
        # ESAgent mit korrekter Netzwerkarchitektur instanziieren
        agent_instance = ESAgent(
            observation_space, action_space, seed=config.SEED,
            # Trainingsparameter hier weniger relevant, aber Netzwerkstruktur muss stimmen
            fc1_units=config.ES_FC1_UNITS, 
            fc2_units=config.ES_FC2_UNITS
        )
        try: agent_instance.load(config.ES_MODEL_PATH)
        except FileNotFoundError: return None
        print(f"ES Agent (beste Policy) geladen von {config.ES_MODEL_PATH}.")
            
    return agent_instance # Geladenen Agenten zurückgeben

def main():
    """Hauptfunktion: Lädt alle konfigurierten Agenten, führt Evaluierungsläufe durch und gibt Ergebnisse aus/speichert sie."""
    config.ensure_model_dir_exists() # Stellt sicher, dass das model/ Verzeichnis existiert
    setup_seeds() # Setzt alle Zufallsgeneratoren-Seeds

    # Evaluierungsumgebung erstellen (Rendermodus aus config.py)
    eval_env = gym.make(config.ENV_ID, render_mode=config.RENDER_MODE_EVAL)
    observation_space = eval_env.observation_space # Dimensionen des Beobachtungsraums
    action_space = eval_env.action_space         # Dimensionen des Aktionsraums
    
    evaluation_results = [] # Liste zum Speichern der Statistiken jedes Agenten
    prepared_agents_for_eval = {} # Dictionary zum Speichern der geladenen Agenteninstanzen

    # Alle in config.AGENT_TYPES spezifizierten Agenten versuchen zu laden
    for agent_type_to_load in config.AGENT_TYPES:
        agent = load_agent(agent_type_to_load, observation_space, action_space)
        if agent: # Nur erfolgreich geladene Agenten hinzufügen
            prepared_agents_for_eval[agent_type_to_load] = agent
    
    # Beenden, falls keine Agenten geladen werden konnten
    if not prepared_agents_for_eval:
        print("\nKeine Agenten konnten für die Evaluierung geladen werden. Beende.")
        eval_env.close(); return

    print(f"\n\n--- Finale Evaluierungsphase ---")
    # Iteriere über alle erfolgreich vorbereiteten Agenten
    for agent_name, agent_instance in prepared_agents_for_eval.items():
        print(f"\nEvaluiere Agenten: {agent_name.upper()} für {config.NUM_EVAL_EPISODES} Episoden...")
        agent_rewards = [] # Liste für Belohnungen dieses Agenten
        agent_steps = []   # Liste für Schritte dieses Agenten

        for i_episode in range(config.NUM_EVAL_EPISODES): # Führe N Evaluierungsepisoden durch
            eval_seed = config.SEED + 2000 + i_episode # Eindeutige Seeds für jeden Evaluierungslauf
            
            episode_reward, steps_taken = run_evaluation_episode(
                eval_env, agent_instance, eval_seed
            )
            agent_rewards.append(episode_reward)
            agent_steps.append(steps_taken)
            print(f"  Eval Episode {i_episode + 1}/{config.NUM_EVAL_EPISODES}: Belohnung = {episode_reward:.2f}, Schritte = {steps_taken}")
        
        # Statistiken für diesen Agenten berechnen
        stats = {
            "Agent": agent_name.upper(),
            "Avg Reward": np.mean(agent_rewards) if agent_rewards else 0,
            "Std Reward": np.std(agent_rewards) if agent_rewards else 0,
            "Min Reward": np.min(agent_rewards) if agent_rewards else 0,
            "Max Reward": np.max(agent_rewards) if agent_rewards else 0,
            "Avg Steps": np.mean(agent_steps) if agent_steps else 0,
            "Num Eval Episodes": len(agent_rewards)
        }
        evaluation_results.append(stats) # Statistiken der Liste hinzufügen
    
    eval_env.close() # Evaluierungsumgebung schließen

    # --- Evaluierungszusammenfassung ausgeben und speichern ---
    if not evaluation_results:
        print("\nKeine Evaluierungsergebnisse zum Anzeigen vorhanden."); return

    print("\n\n--- Evaluierungszusammenfassung (Tabelle) ---")
    # Spaltenüberschriften für die Tabelle und CSV
    headers = ["Agent", "Avg Reward", "Std Reward", "Min Reward", "Max Reward", "Avg Steps", "Num Eval Episodes"]
    
    # Spaltenbreiten für eine schönere Konsolenausgabe dynamisch bestimmen
    col_widths = {header: len(header) for header in headers}
    for row_data in evaluation_results:
        for header in headers:
            value = row_data.get(header) # .get() für sicheren Zugriff
            if value is not None:
                 col_widths[header] = max(col_widths[header], len(f"{value:.2f}" if isinstance(value, float) else str(value)))
            else: # Falls ein Wert fehlt (sollte nicht passieren)
                 col_widths[header] = max(col_widths[header], len("N/A"))

    # Kopfzeile der Tabelle ausgeben
    header_row_str = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    print(header_row_str)
    print("-" * len(header_row_str)) # Trennlinie

    # Datenzeilen der Tabelle ausgeben
    for row_data in evaluation_results:
        row_str_parts = []
        for h in headers:
            value = row_data.get(h)
            # Floats formatieren, ansonsten als String, linksbündig mit Spaltenbreite
            formatted_value = (f"{value:.2f}" if isinstance(value, float) else str(value)) if value is not None else "N/A"
            row_str_parts.append(f"{formatted_value:<{col_widths[h]}}")
        row_str = " | ".join(row_str_parts)
        print(row_str)

    # Ergebnisse in CSV-Datei speichern
    try:
        with open(config.EVALUATION_CSV_PATH, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader() # Schreibt die Kopfzeile in die CSV
            for row_data in evaluation_results: # Schreibt die Datenzeilen
                formatted_row = {} # Dictionary für die CSV-Zeile erstellen
                for h in headers: # Werte formatieren für Konsistenz
                    value = row_data.get(h)
                    formatted_row[h] = (f"{value:.2f}" if isinstance(value, float) else value) if value is not None else "N/A"
                writer.writerow(formatted_row)
        print(f"\nEvaluierungszusammenfassung gespeichert unter: {config.EVALUATION_CSV_PATH}")
    except IOError:
        print(f"\nFEHLER: Konnte Evaluierungszusammenfassung nicht unter {config.EVALUATION_CSV_PATH} schreiben.")

if __name__ == "__main__":
    main()