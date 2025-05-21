# lunar_lander_agents/agents/a2c_agent.py
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

class ActorCriticNetwork(nn.Module): # Erbt von PyTorch's nn.Module
    """
    Definiert ein neuronales Netzwerk, das sowohl eine Policy (Actor) als auch eine
    Value-Funktion (Critic) approximiert. Kann gemeinsame untere Schichten haben.
    """
    def __init__(self, state_size, action_size, seed, 
                 fc1_units=config.A2C_FC1_UNITS, # Nutzt Architekturparameter aus config.py
                 fc2_units=config.A2C_FC2_UNITS):
        super(ActorCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed) # Setzt den PyTorch-Seed für diese Netzwerkinstanz
        
        # Optionale gemeinsame (shared) Schichten für Actor und Critic
        self.fc_shared1 = nn.Linear(state_size, fc1_units)
        self.fc_shared2 = nn.Linear(fc1_units, fc2_units)

        # Actor-spezifischer "Kopf": Gibt Logits für Aktionen aus
        self.actor_head = nn.Linear(fc2_units, action_size) 

        # Critic-spezifischer "Kopf": Gibt den geschätzten Zustandswert V(s) aus (ein einzelner Wert)
        self.critic_head = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Definiert den Forward-Pass des Netzwerks."""
        # Forward-Pass durch die gemeinsamen Schichten
        x = F.relu(self.fc_shared1(state))
        x = F.relu(self.fc_shared2(x))
        
        # Output des Actor-Kopfes (Aktions-Logits)
        action_logits = self.actor_head(x)
        # Output des Critic-Kopfes (Zustandswert)
        state_value = self.critic_head(x)
        
        return action_logits, state_value # Gibt beides zurück

class A2CAgent(BaseAgent): # Erbt von der BaseAgent-Klasse
    """
    Implementiert den Advantage Actor-Critic (A2C) Algorithmus.
    A2C lernt gleichzeitig eine Policy (Actor) und eine Value-Funktion (Critic).
    Updates erfolgen typischerweise nach jedem Schritt (on-policy).
    """
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 seed=0, learning_rate=config.A2C_LEARNING_RATE, # Nutzt Hyperparameter aus config.py
                 gamma=config.A2C_GAMMA, 
                 entropy_coeff=config.A2C_ENTROPY_COEFF, # Koeffizient für den Entropie-Bonus
                 value_loss_coeff=config.A2C_VALUE_LOSS_COEFF, # Gewicht des Critic-Verlusts
                 fc1_units=config.A2C_FC1_UNITS, 
                 fc2_units=config.A2C_FC2_UNITS):
        super().__init__(observation_space, action_space) # Ruft Konstruktor der Basisklasse auf
        
        self.state_size = observation_space.shape[0] # Dimension des Zustandsraums
        self.action_size = action_space.n          # Anzahl der diskreten Aktionen
        self._agent_seed = seed                     # Seed für den Agenten
        self.gamma = gamma                         # Diskontierungsfaktor
        self.lr = learning_rate                    # Lernrate für den Optimierer
        self.entropy_coeff = entropy_coeff         # Zur Förderung der Exploration
        self.value_loss_coeff = value_loss_coeff   # Gewichtung des Critic-Verlusts im Gesamtverlust

        # Seeds für Reproduzierbarkeit setzen
        random.seed(self._agent_seed)
        torch.manual_seed(self._agent_seed)
        if config.DEVICE.type == 'cuda': # Falls CUDA (GPU) verfügbar ist
            torch.cuda.manual_seed_all(self._agent_seed)

        # Actor-Critic Netzwerk initialisieren und auf das konfigurierte Gerät verschieben
        self.network = ActorCriticNetwork(
            self.state_size, self.action_size, self._agent_seed, fc1_units, fc2_units
        ).to(config.DEVICE)
        
        # Adam-Optimierer für das Training des gemeinsamen Netzwerks (oder separater Optimierer, falls separate Netze)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # A2C ist on-policy. Diese einfache Implementierung führt Updates nach jedem Schritt durch.
        # Fortgeschrittenere Versionen sammeln N-Schritt-Returns oder verwenden GAE.

        print(f"A2C Agent initialisiert. LR: {self.lr}, Gamma: {self.gamma}, Gerät: {config.DEVICE}")

    def select_action(self, state, return_log_prob_and_entropy=False): # Optional: Rückgabe von log_prob und Entropie
        """Wählt eine Aktion basierend auf der Policy des Actors."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(config.DEVICE) # Zustand in Tensor umwandeln
        
        # Netzwerk temporär in den Evaluationsmodus für die Inferenz (beeinflusst z.B. kein Dropout)
        self.network.eval() 
        with torch.no_grad(): # Keine Gradientenberechnung für reine Aktionsauswahl nötig
            action_logits, _ = self.network(state_tensor) # Nur Aktions-Logits werden für die Auswahl benötigt
        self.network.train() # Zurück in den Trainingsmodus für nachfolgende Lernschritte

        action_probs = F.softmax(action_logits, dim=-1) # Logits in Wahrscheinlichkeiten umwandeln
        m = Categorical(action_probs) # Kategoriale Verteilung erstellen
        action = m.sample()           # Aktion aus der Verteilung sampeln
        
        # Optional: Gibt auch Log-Wahrscheinlichkeit und Entropie zurück (nützlich für manche A2C-Varianten)
        if return_log_prob_and_entropy:
            return action.item(), m.log_prob(action), m.entropy()
        return action.item() # Gibt die Aktion als Python-Integer zurück

    def learn(self, state, action, reward, next_state, terminated, truncated):
        """Führt einen Lernschritt für Actor und Critic durch."""
        done = terminated or truncated # Kombiniertes Beendigungsflag
        
        # Daten in PyTorch-Tensoren umwandeln und auf das richtige Gerät verschieben
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(config.DEVICE)
        action_tensor = torch.tensor([action], dtype=torch.int64).unsqueeze(0).to(config.DEVICE) # Aktion als LongTensor für gather/log_prob
        reward_tensor = torch.tensor([reward], dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(config.DEVICE)
        done_tensor = torch.tensor([done], dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

        # Forward-Pass: Aktions-Logits und aktuellen Zustandswert vom Netzwerk erhalten
        action_logits, current_state_value = self.network(state_tensor)
        
        # Wert des nächsten Zustands V(S_{t+1}) vom Critic schätzen lassen
        # .detach() wird hier nicht benötigt, da es im with torch.no_grad() Block ist.
        with torch.no_grad(): # Zielwerte sollten nicht zum Gradienten des Critics beitragen
            _, next_state_value = self.network(next_state_tensor) # Nur der Wert des nächsten Zustands ist relevant
        
        # Berechnung des Zielwerts für den Critic (TD-Target): R_t + gamma * V(S_{t+1})
        # Wenn die Episode beendet ist (done=1), ist der Wert des nächsten Zustands 0.
        target_value = reward_tensor + self.gamma * next_state_value * (1 - done_tensor)

        # Berechnung des Advantage A_t = (R_t + gamma * V(S_{t+1})) - V(S_t)
        # Dies ist der TD-Error, der als Advantage für den Actor dient.
        advantage = target_value - current_state_value

        # --- Actor (Policy) Verlust berechnen ---
        action_probs = F.softmax(action_logits, dim=-1) # Wahrscheinlichkeiten aus Logits
        m = Categorical(action_probs)                   # Verteilung für Log-Prob und Entropie
        log_prob = m.log_prob(action_tensor.squeeze(-1)) # Log-Wahrscheinlichkeit der ausgeführten Aktion
        
        # Actor-Verlust: -log_prob * Advantage.
        # .detach() beim Advantage, da der Actor-Verlust nicht die Gewichte des Critics beeinflussen soll.
        # .mean() falls Batch-Verarbeitung (hier Batch-Größe 1, daher optional)
        actor_loss = -(log_prob * advantage.detach()).mean() 

        # --- Critic (Value) Verlust berechnen ---
        # Critic-Verlust: Typischerweise Mean Squared Error (MSE) zwischen dem geschätzten aktuellen Wert
        # und dem berechneten Zielwert (target_value).
        # .detach() bei target_value, da es als festes Ziel dient.
        critic_loss = F.mse_loss(current_state_value, target_value.detach())
        
        # --- Entropie-Verlust berechnen (zur Förderung der Exploration) ---
        entropy = m.entropy().mean() # Durchschnittliche Entropie der Aktionsverteilung
        # Ziel ist es, die Entropie zu maximieren, daher negativer Koeffizient im Gesamtverlust.
        entropy_loss = -self.entropy_coeff * entropy 

        # --- Gesamtverlust kombinieren ---
        # Der Gesamtverlust ist eine gewichtete Summe aus Actor-, Critic- und Entropie-Verlust.
        total_loss = actor_loss + self.value_loss_coeff * critic_loss + entropy_loss

        # --- Optimierungsschritt ---
        self.optimizer.zero_grad() # Gradienten zurücksetzen
        total_loss.backward()      # Backpropagation für den Gesamtverlust
        # Optional: Gradient Clipping zur Stabilisierung
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5) 
        self.optimizer.step()      # Netzwerkparameter aktualisieren

    def reset(self):
        """A2C ist on-policy; typischerweise keine spezielle Reset-Logik pro Episode für den Agenten selbst."""
        pass

    def save(self, filename):
        """Speichert die Gewichte des Actor-Critic-Netzwerks."""
        os.makedirs(os.path.dirname(filename), exist_ok=True) # Stellt sicher, dass das Verzeichnis existiert
        torch.save(self.network.state_dict(), filename) # Speichert den state_dict
        print(f"A2C Agent gespeichert unter: {filename}")

    def load(self, filename):
        """Lädt Gewichte für das Actor-Critic-Netzwerk."""
        if os.path.exists(filename):
            self.network.load_state_dict(torch.load(filename, map_location=config.DEVICE))
            self.network.eval() # Nach dem Laden in den Evaluationsmodus setzen
            print(f"A2C Agent geladen von: {filename}")
        else:
            print(f"FEHLER: Keine A2C Agenten-Datei unter {filename} gefunden.")
            raise FileNotFoundError(f"A2C Modelldatei nicht gefunden: {filename}")