# lunar_lander_agents/agents/reinforce_agent.py
import torch
import torch.nn as nn # PyTorch-Modul für neuronale Netze
import torch.optim as optim # PyTorch-Modul für Optimierungsalgorithmen
import torch.nn.functional as F # PyTorch-Funktionen wie Aktivierungsfunktionen
from torch.distributions import Categorical # Für die kategoriale Verteilung zur Aktionsauswahl
import numpy as np
import random
from .base_agent import BaseAgent # Basisklasse für Agenten
import gymnasium as gym
import os

# Importiert die globale Konfigurationsdatei
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Fügt das Projekt-Stammverzeichnis zum Python-Pfad hinzu
import config

class PolicyNetworkREINFORCE(nn.Module): # Erbt von PyTorch's nn.Module
    """
    Definiert die Architektur des neuronalen Netzwerks, das als Policy für den REINFORCE-Agenten dient.
    Nimmt einen Zustand als Eingabe und gibt Rohwerte (Scores/Logits) für jede Aktion aus.
    """
    def __init__(self, state_size, action_size, seed, 
                 fc1_units=config.REINFORCE_FC1_UNITS, # Nutzt Architekturparameter aus config.py
                 fc2_units=config.REINFORCE_FC2_UNITS):
        super(PolicyNetworkREINFORCE, self).__init__()
        self.seed = torch.manual_seed(seed) # Setzt den PyTorch-Seed für diese Netzwerkinstanz
        # Definition der linearen Schichten (fully connected layers)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size) # Ausgabeschicht gibt Logits für jede Aktion

    def forward(self, state):
        """Definiert den Forward-Pass des Netzwerks."""
        x = F.relu(self.fc1(state)) # ReLU-Aktivierungsfunktion
        x = F.relu(self.fc2(x)) # ReLU-Aktivierungsfunktion
        return self.fc3(x) # Gibt die rohen Aktions-Logits zurück


class REINFORCEAgent(BaseAgent): # Erbt von der BaseAgent-Klasse
    """
    Implementiert den REINFORCE (Monte Carlo Policy Gradient) Algorithmus.
    Lernt eine Policy direkt, indem Aktionen, die zu höheren Returns führen, wahrscheinlicher gemacht werden.
    Updates erfolgen am Ende jeder Episode.
    """
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 seed=0, learning_rate=config.REINFORCE_LEARNING_RATE, # Nutzt Hyperparameter aus config.py
                 gamma=config.REINFORCE_GAMMA, 
                 fc1_units=config.REINFORCE_FC1_UNITS, 
                 fc2_units=config.REINFORCE_FC2_UNITS):
        super().__init__(observation_space, action_space) # Ruft Konstruktor der Basisklasse auf
        
        self.state_size = observation_space.shape[0] # Dimension des Zustandsraums
        self.action_size = action_space.n          # Anzahl der diskreten Aktionen
        self._agent_seed = seed                     # Seed für den Agenten
        self.gamma = gamma                         # Diskontierungsfaktor für zukünftige Belohnungen
        self.lr = learning_rate                    # Lernrate für den Optimierer

        # Seeds für Reproduzierbarkeit setzen
        random.seed(self._agent_seed)
        torch.manual_seed(self._agent_seed)
        if config.DEVICE.type == 'cuda': # Falls CUDA (GPU) verfügbar ist
            torch.cuda.manual_seed_all(self._agent_seed)

        # Policy-Netzwerk initialisieren und auf das konfigurierte Gerät verschieben
        self.policy_network = PolicyNetworkREINFORCE(
            self.state_size, self.action_size, self._agent_seed, fc1_units, fc2_units
        ).to(config.DEVICE)
        
        # Adam-Optimierer für das Training des Policy-Netzwerks
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)

        # Listen zum Speichern der Log-Wahrscheinlichkeiten der Aktionen und der Belohnungen einer Episode
        self.saved_log_probs = [] # Speichert Tensoren mit grad_fn
        self.rewards = []         # Speichert numerische Belohnungswerte

        print(f"REINFORCE Agent initialisiert. Zustandsgröße: {self.state_size}, Aktionsgröße: {self.action_size}, LR: {self.lr}, Gamma: {self.gamma}, Gerät: {config.DEVICE}")

    def select_action(self, state):
        """Wählt eine Aktion basierend auf der Policy und speichert deren Log-Wahrscheinlichkeit."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(config.DEVICE) # Zustand in Tensor umwandeln
        
        # Netzwerk in den Trainingsmodus versetzen, um sicherzustellen, dass der Computation Graph
        # für die Berechnung der Log-Wahrscheinlichkeiten korrekt aufgebaut wird.
        self.policy_network.train() 
        action_logits = self.policy_network(state_tensor) # Logits vom Netzwerk erhalten
        
        action_probs = F.softmax(action_logits, dim=-1) # Logits in Wahrscheinlichkeiten umwandeln
        m = Categorical(action_probs) # Kategoriale Verteilung aus den Wahrscheinlichkeiten erstellen
        action = m.sample()           # Aktion aus der Verteilung sampeln
        
        # Log-Wahrscheinlichkeit der gesampelten Aktion speichern. Dieser Tensor behält seinen grad_fn.
        log_prob_for_this_action = m.log_prob(action) 
        self.saved_log_probs.append(log_prob_for_this_action)
        
        return action.item() # Gibt die Aktion als Python-Integer zurück

    def store_reward(self, reward):
        """Speichert die erhaltene Belohnung für die aktuelle Episode."""
        self.rewards.append(reward)

    def learn_episode(self):
        """Führt den Lernupdate am Ende einer abgeschlossenen Episode durch."""
        # Nichts tun, wenn keine Log-Wahrscheinlichkeiten gespeichert wurden (sollte nicht passieren bei >0 Schritten)
        if not self.saved_log_probs:
            self.reset_episode_data() # Trotzdem Daten zurücksetzen
            return

        # Berechne die diskontierten Returns (G_t) für jeden Zeitschritt der Episode
        discounted_returns = []
        R = 0 # Kumulativer diskontierter Return, beginnend vom Ende der Episode
        for r in reversed(self.rewards): # Iteriere rückwärts durch die Belohnungen
            R = r + self.gamma * R # Bellman-Gleichung für G_t
            discounted_returns.insert(0, R) # Füge G_t am Anfang der Liste ein, um die Reihenfolge beizubehalten

        # Konvertiere Returns in einen Tensor und normalisiere sie (optional, aber oft hilfreich zur Stabilisierung)
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32).to(config.DEVICE)
        if len(discounted_returns) > 1: # Normalisierung nur sinnvoll bei mehr als einem Wert
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-9) # 1e-9 zur Vermeidung von Division durch Null

        policy_loss_components = [] # Liste für die einzelnen Verlustkomponenten
        # Iteriere über die gespeicherten Log-Wahrscheinlichkeiten und die zugehörigen diskontierten Returns
        for log_prob, G_t in zip(self.saved_log_probs, discounted_returns):
            # Verlustterm: -log_pi(a|s) * G_t. Das negative Vorzeichen, da Optimierer minimieren.
            # G_t ist hier ein skalarer Tensor (konstant für diesen Verlustterm).
            # Der Gradient fließt durch log_prob zurück zu den Netzwerkparametern.
            policy_loss_components.append(-log_prob * G_t)

        if not policy_loss_components: # Sollte nicht passieren, wenn self.saved_log_probs nicht leer war
            self.reset_episode_data()
            return

        self.optimizer.zero_grad() # Gradienten des Optimierers zurücksetzen
        # Stapele die einzelnen Verlustkomponenten zu einem Tensor und summiere sie
        policy_loss_tensor = torch.stack(policy_loss_components).sum()
        
        # Sicherheitscheck: Hat der Verlusttensor einen Gradienten?
        if not policy_loss_tensor.requires_grad:
            print("KRITISCHE WARNUNG (REINFORCE): policy_loss_tensor benötigt keinen Gradienten. Update wird übersprungen.")
            self.reset_episode_data()
            return

        policy_loss_tensor.backward() # Backpropagation: Gradienten berechnen
        self.optimizer.step()         # Optimierungsschritt: Netzwerkparameter aktualisieren

        self.reset_episode_data() # Daten für die nächste Episode zurücksetzen

    def learn(self, observation, action, reward, next_observation, terminated, truncated):
        """
        Diese Methode wird vom Runner nach jedem Schritt aufgerufen.
        Für REINFORCE speichert sie nur die Belohnung. Das eigentliche Lernen
        erfolgt in `learn_episode()` am Ende der Episode.
        """
        self.store_reward(reward)

    def reset_episode_data(self):
        """Setzt die Listen für Log-Wahrscheinlichkeiten und Belohnungen zurück."""
        self.saved_log_probs = []
        self.rewards = []

    def reset(self):
        """Wird zu Beginn jeder neuen Episode aufgerufen. Setzt episodenspezifische Daten zurück."""
        self.reset_episode_data()

    def save(self, filename):
        """Speichert die Gewichte des Policy-Netzwerks."""
        os.makedirs(os.path.dirname(filename), exist_ok=True) # Stellt sicher, dass das Verzeichnis existiert
        torch.save(self.policy_network.state_dict(), filename) # Speichert den state_dict (Parameter)
        print(f"REINFORCE Agent gespeichert unter: {filename}")

    def load(self, filename):
        """Lädt Gewichte für das Policy-Netzwerk."""
        if os.path.exists(filename):
            self.policy_network.load_state_dict(torch.load(filename, map_location=config.DEVICE))
            self.policy_network.eval() # Nach dem Laden in den Evaluationsmodus setzen
            print(f"REINFORCE Agent geladen von: {filename}")
        else:
            print(f"FEHLER: Keine REINFORCE Agenten-Datei unter {filename} gefunden.")
            raise FileNotFoundError(f"REINFORCE Modelldatei nicht gefunden: {filename}")