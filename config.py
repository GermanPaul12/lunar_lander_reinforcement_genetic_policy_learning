# lunar_lander_agents/config.py
import os
import torch

# --- Allgemeine Umgebungs- & Projektkonfiguration ---
ENV_ID = "LunarLander-v3" # ID der Gymnasium-Umgebung
SEED = 42                 # Globaler Seed für Reproduzierbarkeit
PROJECT_ROOT = "."        # Projekt-Stammverzeichnis (Annahme: Skripte werden von hier ausgeführt)

# --- Agententypen ---
# Liste der Agententypen, die von Trainings- und Evaluierungsskripten verarbeitet werden
AGENT_TYPES = ["random", "dqn", "genetic", "reinforce", "a2c", "ppo", "es"]

# --- Modellpfade & Verzeichnisse ---
# Stellt sicher, dass das 'models'-Verzeichnis existiert (Skripte versuchen es zu erstellen)
MODEL_DIR = f"{PROJECT_ROOT}/models" # Hauptverzeichnis für gespeicherte Modelle

# Dateinamen und vollständige Pfade für die Modelle der einzelnen Agenten
DQN_MODEL_FILENAME = "dqn_lunar_lander.pth"
DQN_MODEL_PATH = f"{MODEL_DIR}/{DQN_MODEL_FILENAME}"

GA_MODEL_FILENAME = "ga_best_lunar_lander.pth"
GA_MODEL_PATH = f"{MODEL_DIR}/{GA_MODEL_FILENAME}"

REINFORCE_MODEL_FILENAME = "reinforce_lunar_lander.pth"
REINFORCE_MODEL_PATH = f"{MODEL_DIR}/{REINFORCE_MODEL_FILENAME}"

A2C_MODEL_FILENAME = "a2c_lunar_lander.pth"
A2C_MODEL_PATH = f"{MODEL_DIR}/{A2C_MODEL_FILENAME}"

PPO_ACTOR_MODEL_FILENAME = "ppo_actor_lunar_lander.pth" # PPO hat separate Modelle für Actor und Critic
PPO_CRITIC_MODEL_FILENAME = "ppo_critic_lunar_lander.pth"
PPO_ACTOR_MODEL_PATH = f"{MODEL_DIR}/{PPO_ACTOR_MODEL_FILENAME}"
PPO_CRITIC_MODEL_PATH = f"{MODEL_DIR}/{PPO_CRITIC_MODEL_FILENAME}"

ES_MODEL_FILENAME = "es_lunar_lander.pth"
ES_MODEL_PATH = f"{MODEL_DIR}/{ES_MODEL_FILENAME}"

# Dateiname und Pfad für die CSV-Datei mit den Evaluierungsergebnissen
EVALUATION_CSV_FILENAME = "evaluation_summary.csv"
EVALUATION_CSV_PATH = f"{MODEL_DIR}/{EVALUATION_CSV_FILENAME}"

# --- Trainingskonfiguration ---
# Flags, um ein Neutraining zu erzwingen, auch wenn eine Modelldatei existiert
FORCE_RETRAIN_ALL = False # Globaler Schalter, um alle Agenten neu zu trainieren
FORCE_RETRAIN_DQN = False or FORCE_RETRAIN_ALL
FORCE_RETRAIN_GA = False or FORCE_RETRAIN_ALL
FORCE_RETRAIN_REINFORCE = False or FORCE_RETRAIN_ALL
FORCE_RETRAIN_A2C = False or FORCE_RETRAIN_ALL
FORCE_RETRAIN_PPO = False or FORCE_RETRAIN_ALL
FORCE_RETRAIN_ES = False or FORCE_RETRAIN_ALL

# === DQN Trainingsparameter ===
DQN_TRAIN_EPISODES = 10000       # Anzahl der Trainingsepisoden für DQN
DQN_MAX_T_PER_EPISODE = 1000    # Maximale Schritte pro DQN-Trainingsepisode
DQN_SCORE_TARGET = 200.0        # Ziel-Score, bei dem das DQN-Training als "gelöst" gilt
DQN_PRINT_EVERY = 100           # Fortschritt alle N Episoden ausgeben

# DQN Hyperparameter
DQN_BUFFER_SIZE = int(1e5)      # Größe des Replay Buffers
DQN_BATCH_SIZE = 64             # Minibatch-Größe für Trainingsupdates
DQN_GAMMA = 0.99                # Diskontierungsfaktor für zukünftige Belohnungen
DQN_LR = 5e-4                   # Lernrate für den Adam-Optimierer
DQN_UPDATE_EVERY = 4            # Netzwerkupdate alle N Schritte
DQN_TARGET_UPDATE_EVERY = 100   # Update des Target-Netzwerks alle N Lernschritte
DQN_EPSILON_START = 1.0         # Startwert für Epsilon (Exploration)
DQN_EPSILON_MIN = 0.01          # Minimalwert für Epsilon
DQN_EPSILON_DECAY = 0.995       # Multiplikativer Zerfallsfaktor für Epsilon

# === Genetischer Algorithmus (GA) Trainingsparameter ===
GA_N_GENERATIONS = 10000           # Anzahl der Generationen für GA
GA_POPULATION_SIZE = 200         # Anzahl der Individuen in der GA-Population
GA_EVAL_EPISODES_PER_INDIVIDUAL = 2 # Episoden zur Fitness-Evaluierung jedes GA-Individuums
GA_MAX_STEPS_PER_GA_EVAL = 500    # Max. Schritte während der GA-Fitness-Evaluierung
GA_SAVE_EVERY_N_GENERATIONS = 10  # Speichere das beste GA-Modell alle N Generationen

# GA Hyperparameter
GA_MUTATION_RATE = 0.1          # Wahrscheinlichkeit einer Mutation
GA_MUTATION_STRENGTH = 0.1      # Stärke der Mutation (z.B. Standardabweichung des Rauschens)
GA_CROSSOVER_RATE = 0.7         # Wahrscheinlichkeit für Crossover zwischen Eltern
GA_TOURNAMENT_SIZE = 5          # Größe der Turnierauswahl für die Selektion
GA_ELITISM_COUNT = 2            # Anzahl der besten Individuen, die direkt in die nächste Generation übernommen werden

# GA Policy Netzwerkarchitektur
GA_FC1_UNITS = 64               # Anzahl Neuronen in der ersten versteckten Schicht des GA-Policy-Netzwerks
GA_FC2_UNITS = 32               # Anzahl Neuronen in der zweiten versteckten Schicht

# === REINFORCE Trainingsparameter ===
REINFORCE_TRAIN_EPISODES = 20000 # REINFORCE benötigt oft mehr Episoden aufgrund hoher Varianz
REINFORCE_MAX_T_PER_EPISODE = 1000
REINFORCE_PRINT_EVERY = 100
REINFORCE_SCORE_TARGET = 150.0  # Ziel-Score kann niedriger sein oder länger dauern ohne Baseline

# REINFORCE Hyperparameter
REINFORCE_LEARNING_RATE = 1e-3
REINFORCE_GAMMA = 0.99
# REINFORCE Policy Netzwerkarchitektur
REINFORCE_FC1_UNITS = 128
REINFORCE_FC2_UNITS = 64

# === A2C (Advantage Actor-Critic) Trainingsparameter ===
A2C_TRAIN_EPISODES = 20000
A2C_MAX_T_PER_EPISODE = 1000
A2C_PRINT_EVERY = 100
A2C_SCORE_TARGET = 180.0

# A2C Hyperparameter
A2C_LEARNING_RATE = 7e-4        # Oft etwas höher für A2C
A2C_GAMMA = 0.99
A2C_ENTROPY_COEFF = 0.01        # Gewicht für den Entropie-Bonus zur Förderung der Exploration
A2C_VALUE_LOSS_COEFF = 0.5      # Gewicht für den Value-Loss im Gesamtverlust
# A2C Netzwerkarchitektur (kann geteilt oder separat sein)
A2C_FC1_UNITS = 128
A2C_FC2_UNITS = 64

# === PPO (Proximal Policy Optimization) Trainingsparameter ===
PPO_TOTAL_TIMESTEPS = 10_000_000 # PPO wird oft über eine Gesamtanzahl von Zeitschritten trainiert
PPO_UPDATE_HORIZON = 2048       # Anzahl der Schritte, die gesammelt werden, bevor eine PPO-Update-Phase erfolgt (Rollout-Länge)
PPO_EPOCHS = 10                # Anzahl der Optimierungsepochen über die gesammelten Daten
PPO_BATCH_SIZE = 64             # Minibatch-Größe für SGD während der PPO-Epochen
PPO_PRINT_EVERY_N_UPDATES = 10  # Fortschritt nach N PPO-Update-Zyklen ausgeben
PPO_SCORE_TARGET = 280.0        # PPO kann hohe Scores erreichen

# PPO Hyperparameter
PPO_ACTOR_LR = 3e-4             # Lernrate für den Actor
PPO_CRITIC_LR = 1e-3            # Lernrate für den Critic
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95           # Lambda für Generalized Advantage Estimation (GAE)
PPO_CLIP_EPSILON = 0.2          # PPO Clipping-Parameter für die Surrogate Objective Function
PPO_ENTROPY_COEFF = 0.005
PPO_VALUE_LOSS_COEFF = 0.5
# PPO Netzwerkarchitekturen
PPO_ACTOR_FC1 = 64
PPO_ACTOR_FC2 = 64
PPO_CRITIC_FC1 = 64
PPO_CRITIC_FC2 = 64

# === Evolutionäre Strategien (ES) Trainingsparameter ===
ES_N_GENERATIONS = 100000          # Anzahl der ES-"Generationen" oder Update-Schritte
ES_POPULATION_SIZE = 200         # Anzahl der Perturbationen (Variationen) pro Generation
ES_SIGMA = 0.1                  # Standardabweichung des Gaußschen Rauschens für Perturbationen
ES_LEARNING_RATE = 0.01         # Lernrate zur Aktualisierung der zentralen Policy-Parameter
ES_EVAL_EPISODES_PER_PARAM = 1  # Episoden zur Fitness-Evaluierung jedes perturbierten Parametersatzes
ES_MAX_T_PER_EVAL_EPISODE = 1000 # Max. Schritte während der ES-Fitness-Evaluierung
ES_PRINT_EVERY = 10
ES_SCORE_TARGET = 220.0

# ES Netzwerkarchitektur
ES_FC1_UNITS = 64
ES_FC2_UNITS = 32

# --- Testkonfiguration (für test.py) ---
NUM_TEST_RUNS = 10              # Anzahl der Testdurchläufe für einen einzelnen Agenten
RENDER_MODE_TEST = "rgb_array"      # Rendermodus für test.py (immer 'human' für Visualisierung)

# --- Evaluierungskonfiguration (für evaluate.py) ---
NUM_EVAL_EPISODES = 1000          # Anzahl der Episoden für die finale Evaluierung jedes Agenten
MAX_STEPS_PER_EVAL_EPISODE = 1000 # Maximale Schritte pro Evaluierungsepisode
RENDER_MODE_EVAL = None         # Rendermodus für evaluate.py ('human' zum Zuschauen, None für schnellere headless Läufe)
# RENDER_MODE_EVAL = "human" # Alternative: Evaluierung mit Visualisierung

# --- Gerätekonfiguration (CPU/GPU) ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Automatische Auswahl von GPU (CUDA) falls verfügbar

# --- Hilfsfunktion zum Erstellen des Modellverzeichnisses ---
def ensure_model_dir_exists():
    """Stellt sicher, dass das Verzeichnis zum Speichern der Modelle existiert."""
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs(MODEL_DIR)
            print(f"Verzeichnis erstellt: {MODEL_DIR}")
        except OSError as e:
            print(f"Fehler beim Erstellen des Verzeichnisses {MODEL_DIR}: {e}")
            # Je nach Schweregrad könnte hier ein Abbruch oder das Auslösen der Exception sinnvoll sein