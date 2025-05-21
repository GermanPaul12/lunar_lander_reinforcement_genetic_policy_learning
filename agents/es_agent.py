# lunar_lander_agents/agents/es_agent.py
import torch
import torch.nn as nn # PyTorch-Modul für neuronale Netze
import torch.nn.functional as F # PyTorch-Funktionen wie Aktivierungsfunktionen
import numpy as np
import random
from .base_agent import BaseAgent # Basisklasse für Agenten
import gymnasium as gym
import os
# from collections import deque # deque wird hier nicht direkt verwendet

# Importiert die globale Konfigurationsdatei
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

# Policy-Netzwerkstruktur, die für ES verwendet wird. Kann ähnlich wie bei GA oder REINFORCE sein.
class PolicyNetworkES(nn.Module): # Erbt von PyTorch's nn.Module
    """
    Definiert die Architektur des neuronalen Netzwerks, dessen Parameter durch
    Evolutionäre Strategien optimiert werden.
    """
    def __init__(self, state_size, action_size, seed, 
                 fc1_units=config.ES_FC1_UNITS, # Nutzt Architekturparameter aus config.py
                 fc2_units=config.ES_FC2_UNITS):
        super(PolicyNetworkES, self).__init__()
        # Wichtig: Der Seed hier initialisiert die Netzwerkstruktur konsistent,
        # wenn nur die Gewichte evolviert werden.
        self.seed = torch.manual_seed(seed) 
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size) # Gibt Aktions-Logits aus

    def forward(self, state):
        """Definiert den Forward-Pass des Netzwerks."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x) # Gibt rohe Aktions-Logits zurück

    def get_weights_flat(self):
        """Hilfsmethode, um alle Gewichte als einen flachen NumPy-Array zu erhalten."""
        # Nützlich für die Parameter-Perturbation in ES.
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

    def set_weights_flat(self, flat_weights):
        """Hilfsmethode, um Gewichte aus einem flachen NumPy-Array zu setzen."""
        offset = 0
        for param in self.parameters(): # Iteriert über alle Parameter des Netzwerks
            shape = param.data.shape
            num_elements = param.data.numel()
            # Kopiert den entsprechenden Teil des flachen Arrays in den Parameter-Tensor
            param.data.copy_(torch.from_numpy(flat_weights[offset:offset+num_elements]).view(shape).to(config.DEVICE))
            offset += num_elements
        if offset != len(flat_weights): # Sicherheitscheck für korrekte Gewichtsgröße
            raise ValueError(f"Größe von flat_weights ({len(flat_weights)}) stimmt nicht mit Modellparametern ({offset} erwartet) überein.")


class ESAgent(BaseAgent): # Erbt von BaseAgent. Diese Klasse agiert als Controller für den ES-Prozess.
    """
    Implementiert einen Evolutionäre Strategien (ES) Agenten.
    ES ist eine direkte Policy-Suchmethode, die Parameter einer Policy (hier eines neuronalen Netzes)
    durch iterative Perturbation und Selektion basierend auf der Fitness optimiert.
    Der ESAgent selbst hält die beste gefundene Policy und nutzt diese für Aktionen.
    """
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 seed=0, population_size=config.ES_POPULATION_SIZE, # Anzahl der Perturbationen
                 sigma=config.ES_SIGMA,                     # Standardabweichung des Rauschens
                 learning_rate=config.ES_LEARNING_RATE,     # Lernrate für Update der zentralen Parameter
                 eval_episodes_per_param=config.ES_EVAL_EPISODES_PER_PARAM, # Episoden zur Fitness-Evaluation
                 fc1_units=config.ES_FC1_UNITS, # Architekturparameter für das Policy-Netzwerk
                 fc2_units=config.ES_FC2_UNITS):
        super().__init__(observation_space, action_space) # Ruft Konstruktor der Basisklasse auf
        
        # Speichert wichtige Konfigurations- und Hyperparameter
        self.state_size = observation_space.shape[0]
        self.action_size = action_space.n
        self._agent_seed = seed # Seed für den ES-Prozess selbst (z.B. Rauscherzeugung)
        self.population_size = population_size 
        self.sigma = sigma 
        self.learning_rate = learning_rate 
        self.eval_episodes_per_param = eval_episodes_per_param

        # Seeds für Reproduzierbarkeit setzen
        random.seed(self._agent_seed)
        np.random.seed(self._agent_seed) # Wichtig für die Rauscherzeugung
        torch.manual_seed(self._agent_seed) # Seed für die Initialisierung des zentralen Policy-Netzwerks
        if config.DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(self._agent_seed)

        # Zentrales Policy-Netzwerk, dessen Parameter optimiert werden
        self.central_policy_net = PolicyNetworkES(
            self.state_size, self.action_size, self._agent_seed, fc1_units, fc2_units
        ).to(config.DEVICE)
        # Anzahl der trainierbaren Parameter im Netzwerk (wichtig für Rauscherzeugung)
        self.num_params = sum(p.numel() for p in self.central_policy_net.parameters())
        
        # Speichert die Fitness und Gewichte der bisher besten gefundenen Policy
        self.current_best_fitness = -float('inf')
        self.current_best_weights = self.central_policy_net.get_weights_flat().copy() # Startet mit initialen Gewichten

        print(f"ES Agent (Controller) initialisiert. Pop_Größe: {self.population_size}, Sigma: {self.sigma}, LR: {self.learning_rate}, Anz_Params: {self.num_params}")

    def _evaluate_parameters(self, flat_weights_candidate, eval_env):
        """Evaluiert einen gegebenen Satz von (flachen) Parametern durch Ausführen von Episoden."""
        # Temporäres Policy-Netzwerk erstellen und mit den Kandidaten-Gewichten konfigurieren
        temp_policy_net = PolicyNetworkES(
            self.state_size, self.action_size, seed=0 # Seed hier irrelevant, da Gewichte gesetzt werden
            , fc1_units=config.ES_FC1_UNITS, fc2_units=config.ES_FC2_UNITS # Nutzt config für Konsistenz
        ).to(config.DEVICE)
        temp_policy_net.set_weights_flat(flat_weights_candidate)
        temp_policy_net.eval() # In den Evaluationsmodus setzen

        total_rewards = []
        # Führt mehrere Episoden durch, um eine robustere Fitness-Schätzung zu erhalten
        for _ in range(self.eval_episodes_per_param):
            obs, _ = eval_env.reset(seed=random.randint(0, 100000)) # Variabler Seed für jede Evaluation
            episode_reward = 0
            for _ in range(config.ES_MAX_T_PER_EVAL_EPISODE): # Maximale Schritte pro Eval-Episode aus config
                state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(config.DEVICE)
                with torch.no_grad(): # Keine Gradientenberechnung
                    action_logits = temp_policy_net(state_tensor)
                action = torch.argmax(action_logits, dim=-1).item() # Deterministische Aktion
                
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            total_rewards.append(episode_reward)
        return np.mean(total_rewards) # Durchschnittliche Belohnung als Fitness

    def evolve_step(self, eval_env):
        """Führt einen Evolutionsschritt (eine Generation) durch."""
        current_central_weights = self.central_policy_net.get_weights_flat() # Aktuelle zentrale Gewichte
        
        # Erzeugt 'population_size' Rauschvektoren (Perturbationen)
        # Jeder Rauschvektor hat die gleiche Dimension wie die Anzahl der Netzwerkparameter.
        noise_samples = np.random.randn(self.population_size, self.num_params)
        
        fitness_scores = np.zeros(self.population_size) # Array zur Speicherung der Fitnesswerte
        
        # Evaluiert jeden perturbierten Parametersatz
        for i in range(self.population_size):
            # Perturbierte Gewichte = zentrale Gewichte + sigma * Rauschen
            perturbed_weights = current_central_weights + self.sigma * noise_samples[i]
            fitness_scores[i] = self._evaluate_parameters(perturbed_weights, eval_env)

        # Update-Regel für die zentralen Gewichte (basierend auf OpenAI ES)
        # Standardisierung der Fitness-Scores (Subtraktion des Mittelwerts, Division durch Standardabweichung)
        # Dies hilft, die Lernrate besser zu skalieren und macht den Algorithmus robuster.
        if np.std(fitness_scores) > 1e-6: # Vermeidet Division durch Null bei konstanter Fitness
            standardized_fitness = (fitness_scores - np.mean(fitness_scores)) / np.std(fitness_scores)
        else:
            standardized_fitness = np.zeros_like(fitness_scores) # Kein Update, wenn alle Fitnesswerte gleich sind

        # Update-Richtung wird als gewichtete Summe der Rauschvektoren berechnet,
        # wobei die standardisierten Fitness-Scores als Gewichte dienen.
        update_direction = np.dot(noise_samples.T, standardized_fitness)
        
        # Aktualisiert die zentralen Gewichte
        # Formel: theta_{t+1} = theta_t + learning_rate * (1 / (N*sigma)) * sum(F_i * epsilon_i)
        # Hier ist update_direction = sum(F_standardized_i * epsilon_i)
        # Der Term (self.population_size * self.sigma) ist ein Skalierungsfaktor.
        new_central_weights = current_central_weights + (self.learning_rate / (self.population_size * self.sigma)) * update_direction

        self.central_policy_net.set_weights_flat(new_central_weights) # Setzt die neuen zentralen Gewichte

        # Verfolgt die Fitness der besten jemals gefundenen zentralen Policy
        current_eval_of_central = self._evaluate_parameters(new_central_weights, eval_env) # Evaluiert die neuen zentralen Gewichte
        if current_eval_of_central > self.current_best_fitness:
            self.current_best_fitness = current_eval_of_central
            self.current_best_weights = new_central_weights.copy() # Speichert eine Kopie der besten Gewichte
            print(f"    ES: Neuer bester Fitnesswert für zentrale Policy: {self.current_best_fitness:.2f}")
        
        # Gibt Durchschnitts- und Max-Fitness der aktuellen Perturbations-Population sowie Fitness der neuen zentralen Policy zurück
        return np.mean(fitness_scores), np.max(fitness_scores), current_eval_of_central

    # Für die Kompatibilität mit BaseAgent: Der ESAgent selbst verwendet seine beste gefundene Policy
    def select_action(self, state):
        """Wählt eine Aktion basierend auf der aktuell besten gefundenen Policy."""
        # Stellt sicher, dass das zentrale Netzwerk die besten Gewichte verwendet
        self.central_policy_net.set_weights_flat(self.current_best_weights) 
        self.central_policy_net.eval() # In den Evaluationsmodus setzen

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(config.DEVICE)
        with torch.no_grad(): # Keine Gradientenberechnung
            action_logits = self.central_policy_net(state_tensor)
        action = torch.argmax(action_logits, dim=-1).item() # Deterministische Aktion
        return action

    def learn(self, observation, action, reward, next_observation, terminated, truncated):
        # Das "Lernen" bei ES geschieht in der `evolve_step()` Methode, nicht pro Umgebungsschritt.
        pass

    def reset(self):
        # Der ES-Controller selbst hat keinen episodenspezifischen Zustand, der zurückgesetzt werden muss.
        pass

    def save(self, filename):
        """Speichert die Gewichte der besten gefundenen Policy."""
        os.makedirs(os.path.dirname(filename), exist_ok=True) # Stellt sicher, dass das Verzeichnis existiert
        # Um den state_dict zu speichern, werden die besten Gewichte zuerst in das Netzwerk geladen
        self.central_policy_net.set_weights_flat(self.current_best_weights)
        torch.save(self.central_policy_net.state_dict(), filename) # Speichert den state_dict
        print(f"ES Agent (beste Policy) gespeichert unter: {filename}")

    def load(self, filename):
        """Lädt Gewichte in die zentrale Policy und setzt sie als die aktuell besten."""
        if os.path.exists(filename):
            self.central_policy_net.load_state_dict(torch.load(filename, map_location=config.DEVICE))
            self.central_policy_net.eval() # In den Evaluationsmodus setzen
            self.current_best_weights = self.central_policy_net.get_weights_flat().copy() # Geladene Gewichte als beste setzen
            # Optional: Fitness des geladenen Modells neu bewerten, um current_best_fitness zu aktualisieren.
            # Dies erfordert eine temporäre Umgebung.
            # eval_env_temp = gym.make(config.ENV_ID) 
            # self.current_best_fitness = self._evaluate_parameters(self.current_best_weights, eval_env_temp)
            # eval_env_temp.close()
            print(f"ES Agent (Policy) geladen von: {filename}. Beste Fitness muss ggf. neu evaluiert werden.")
        else:
            print(f"FEHLER: Keine ES Agenten-Datei unter {filename} gefunden.")
            raise FileNotFoundError(f"ES Modelldatei nicht gefunden: {filename}")