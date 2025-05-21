# lunar_lander_agents/agents/dqn_agent.py
import torch
import torch.nn as nn # PyTorch-Modul für neuronale Netze
import torch.optim as optim # PyTorch-Modul für Optimierungsalgorithmen
import torch.nn.functional as F # PyTorch-Funktionen wie Aktivierungsfunktionen
import numpy as np
import random
from collections import deque, namedtuple # deque für Replay Buffer, namedtuple für Experiences
from .base_agent import BaseAgent # Basisklasse für Agenten
import gymnasium as gym
import os

# Importiert die globale Konfigurationsdatei
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

# Hyperparameter werden aus der config.py-Datei geladen
BUFFER_SIZE = config.DQN_BUFFER_SIZE
BATCH_SIZE = config.DQN_BATCH_SIZE
GAMMA = config.DQN_GAMMA # Diskontierungsfaktor
LR = config.DQN_LR       # Lernrate
UPDATE_EVERY = config.DQN_UPDATE_EVERY # Wie oft das lokale Netzwerk aktualisiert wird
TARGET_UPDATE_EVERY = config.DQN_TARGET_UPDATE_EVERY # Wie oft das Zielnetzwerk aktualisiert wird

device = config.DEVICE # Verwendet das in config.py definierte Gerät (CPU oder CUDA)

class QNetwork(nn.Module): # Erbt von PyTorch's nn.Module
    """
    Neuronales Netzwerkmodell zur Approximation der Q-Werte (Aktionswerte).
    Nimmt einen Zustand als Eingabe und gibt für jede mögliche Aktion einen Q-Wert aus.
    """
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        Initialisiert die Parameter und baut das Modell auf.

        Args:
            state_size (int): Dimension des Zustandsraums.
            action_size (int): Anzahl der möglichen Aktionen.
            seed (int): Seed für die Initialisierung der Gewichte zur Reproduzierbarkeit.
            fc1_units (int): Anzahl der Neuronen in der ersten versteckten Schicht.
            fc2_units (int): Anzahl der Neuronen in der zweiten versteckten Schicht.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed) # Setzt den PyTorch-Seed für diese Netzwerkinstanz
        # Definition der linearen Schichten (fully connected layers)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size) # Ausgabeschicht liefert Q-Werte für jede Aktion

    def forward(self, state):
        """Definiert den Forward-Pass des Netzwerks."""
        x = F.relu(self.fc1(state)) # ReLU-Aktivierungsfunktion nach der ersten Schicht
        x = F.relu(self.fc2(x)) # ReLU-Aktivierungsfunktion nach der zweiten Schicht
        return self.fc3(x) # Gibt die Q-Werte für jede Aktion zurück

class ReplayBuffer:
    """
    Ein Puffer mit fester Größe zum Speichern von Erfahrungstupeln (Experience Tuples).
    Ermöglicht das zufällige Sampeln von Erfahrungen für das Training (Experience Replay).
    """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialisiert ein ReplayBuffer-Objekt.

        Args:
            action_size (int): Dimension des Aktionsraums (nicht direkt für Speicherung verwendet, aber oft nützlich).
            buffer_size (int): Maximale Größe des Puffers.
            batch_size (int): Größe jedes Trainings-Minibatches.
            seed (int): Seed für den Zufallszahlengenerator beim Sampeln.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) # deque: doppelseitige Warteschlange, alte Elemente werden automatisch entfernt
        self.batch_size = batch_size
        # Definiert ein benanntes Tupel für eine einzelne Erfahrung
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed) # Setzt den Seed für das `random` Modul

    def add(self, state, action, reward, next_state, done):
        """Fügt eine neue Erfahrung zum Speicher hinzu."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Zieht zufällig einen Batch von Erfahrungen aus dem Speicher."""
        # Wählt `batch_size` zufällige Erfahrungen aus dem Speicher aus
        experiences = random.sample(self.memory, k=self.batch_size)

        # Konvertiert die Listen der einzelnen Komponenten der Erfahrungen in PyTorch-Tensoren
        # `vstack` stapelt die Zustände/Aktionen etc. vertikal zu einem Batch
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        # `done` Flags werden als uint8 (0 oder 1) und dann als float konvertiert
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Gibt die aktuelle Größe des internen Speichers zurück."""
        return len(self.memory)

class DQNAgent(BaseAgent): # Erbt von der BaseAgent-Klasse
    """Interagiert mit der Umgebung und lernt daraus mittels Deep Q-Learning."""
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, seed=0):
        super().__init__(observation_space, action_space) # Ruft Konstruktor der Basisklasse auf
        self.state_size = observation_space.shape[0] # Dimension des Zustandsraums
        self.action_size = action_space.n          # Anzahl der diskreten Aktionen
        
        # Seeds für Reproduzierbarkeit setzen
        self._agent_seed = seed 
        random.seed(self._agent_seed)
        torch.manual_seed(self._agent_seed) # PyTorch Seed für CPU
        if config.DEVICE.type == 'cuda': # Und für GPU, falls verwendet
            torch.cuda.manual_seed_all(self._agent_seed)


        print(f"DQN Agent initialisiert. Zustandsgröße: {self.state_size}, Aktionsgröße: {self.action_size}, Gerät: {device}")

        # Q-Netzwerke: Ein lokales Netzwerk für die Aktionsauswahl und das Lernen,
        # und ein Zielnetzwerk (Target Network) für stabile Zielwerte.
        # Beide Netzwerke haben die gleiche Architektur.
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed=self._agent_seed).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, seed=self._agent_seed).to(device)
        # Adam-Optimierer für das Training des lokalen Netzwerks
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Initialisiert die Gewichte des Zielnetzwerks mit den Gewichten des lokalen Netzwerks
        self.hard_update(self.qnetwork_local, self.qnetwork_target)

        # Replay Buffer zum Speichern und Wiederverwenden von Erfahrungen
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed=self._agent_seed)
        
        # Zähler für die Update-Frequenzen
        self.t_step = 0             # Zählt Schritte für periodisches Lernen (UPDATE_EVERY)
        self.target_update_step = 0 # Zählt Lernschritte für periodisches Update des Target-Netzwerks

        # Epsilon-Parameter für die Epsilon-Greedy-Explorationsstrategie
        self.epsilon = config.DQN_EPSILON_START     # Startwert für Epsilon
        self.epsilon_min = config.DQN_EPSILON_MIN   # Minimalwert für Epsilon
        self.epsilon_decay = config.DQN_EPSILON_DECAY # Zerfallsrate für Epsilon

    def select_action(self, state, eps=None): # Erlaubt Überschreiben von Epsilon für Evaluierung
        """
        Wählt eine Aktion für den gegebenen Zustand gemäß der aktuellen Policy (Epsilon-Greedy).
        """
        # Wenn kein spezifisches Epsilon übergeben wird, das aktuelle Epsilon des Agenten verwenden
        current_epsilon = eps if eps is not None else self.epsilon

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device) # Zustand in Tensor umwandeln
        
        # Netzwerk in den Evaluationsmodus setzen, um Dropout etc. zu deaktivieren
        self.qnetwork_local.eval() 
        with torch.no_grad(): # Keine Gradientenberechnung für reine Inferenz nötig
            action_values = self.qnetwork_local(state_tensor) # Q-Werte für alle Aktionen erhalten
        self.qnetwork_local.train() # Netzwerk zurück in den Trainingsmodus setzen

        # Epsilon-Greedy Aktionsauswahl
        if random.random() > current_epsilon: # Mit Wahrscheinlichkeit (1-epsilon) greedy handeln (Exploitation)
            return np.argmax(action_values.cpu().data.numpy()) # Aktion mit höchstem Q-Wert wählen
        else: # Mit Wahrscheinlichkeit epsilon zufällig handeln (Exploration)
            return random.choice(np.arange(self.action_size))

    def learn_step(self, state, action, reward, next_state, done):
        """Ein einzelner Lernschritt des Agenten: Erfahrung speichern und ggf. lernen."""
        # Erfahrung im Replay Buffer speichern
        self.memory.add(state, action, reward, next_state, done)

        # Lernen alle `UPDATE_EVERY` Zeitschritte
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Wenn genügend Samples im Speicher vorhanden sind, einen Batch ziehen und lernen
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample() # Zufälligen Minibatch aus dem Speicher ziehen
                self._learn_from_experiences(experiences, GAMMA) # Lernschritt mit dem Batch durchführen

        # Zielnetzwerk alle `TARGET_UPDATE_EVERY` Lernschritte aktualisieren
        # (nur wenn tatsächlich ein Lernschritt stattgefunden hat)
        if len(self.memory) > BATCH_SIZE and self.t_step == 0: 
            self.target_update_step = (self.target_update_step + 1) % TARGET_UPDATE_EVERY
            if self.target_update_step == 0:
                # Hier wird ein "hard update" verwendet: Gewichte des lokalen Netzwerks direkt kopieren
                self.hard_update(self.qnetwork_local, self.qnetwork_target)
                # Alternativ: self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) für sanftere Updates

    def _learn_from_experiences(self, experiences, gamma):
        """
        Aktualisiert die Wertparameter (Gewichte des Q-Netzwerks) anhand eines gegebenen
        Batches von Erfahrungstupeln.
        """
        states, actions, rewards, next_states, dones = experiences # Entpackt den Batch

        # === Berechnung der Ziel-Q-Werte (Target Q-Values) ===
        # Q-Werte für die nächsten Zustände (S') vom Zielnetzwerk (qnetwork_target) erhalten.
        # .detach() verhindert, dass Gradienten durch das Zielnetzwerk fließen.
        # .max(1)[0] wählt den maximalen Q-Wert für jeden nächsten Zustand (greedy Policy für S').
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Berechnet die Ziel-Q-Werte für die aktuellen Zustände (S) nach der Bellman-Gleichung:
        # Q_target = R + gamma * max_a' Q_target(S', a') * (1 - done)
        # (1 - dones) stellt sicher, dass für terminale Zustände der zukünftige Wert 0 ist.
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # === Berechnung der erwarteten Q-Werte (Expected Q-Values) ===
        # Q-Werte für die aktuellen Zustände (S) vom lokalen Netzwerk (qnetwork_local) erhalten.
        # .gather(1, actions) wählt die Q-Werte für die tatsächlich ausgeführten Aktionen aus.
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # === Verlustberechnung und Optimierung ===
        # Berechnet den Mean Squared Error (MSE) Verlust zwischen erwarteten und Ziel-Q-Werten.
        loss = F.mse_loss(Q_expected, Q_targets)
        
        self.optimizer.zero_grad() # Gradienten auf Null setzen vor Backpropagation
        loss.backward()            # Backpropagation: Gradienten des Verlusts berechnen
        # Optional: Gradient Clipping, um explodierende Gradienten zu verhindern
        # torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1) 
        self.optimizer.step()      # Optimierungsschritt: Netzwerkparameter aktualisieren

    def learn(self, observation, action, reward, next_observation, terminated, truncated):
        """ Haupt-Lernmethode, die vom Runner aufgerufen wird. """
        done = terminated or truncated # Kombiniert Beendigungsflags
        self.learn_step(observation, action, reward, next_observation, done) # Ruft den internen Lernschritt auf

    def soft_update(self, local_model, target_model, tau):
        """
        Soft-Update der Modellparameter des Zielnetzwerks.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        (Wird in dieser Implementierung nicht standardmäßig verwendet, aber als Option vorhanden)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        """Hard-Update: Kopiert Gewichte vom lokalen zum Zielnetzwerk. θ_target = θ_local"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def decay_epsilon(self):
        """Verringert Epsilon für das Gleichgewicht zwischen Exploration und Exploitation."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self):
        """Wird zu Beginn einer Episode vom Runner aufgerufen. Für DQN: Epsilon verringern."""
        self.decay_epsilon() # Epsilon für die nächste Episode anpassen
        # print(f"Epsilon zerfallen zu: {self.epsilon:.4f}") # Für Debugging

    def save(self, filename=None):
        """Speichert die Gewichte des lokalen Q-Netzwerks."""
        path = filename if filename else config.DQN_MODEL_PATH # Nutzt Pfad aus config.py wenn keiner übergeben
        os.makedirs(os.path.dirname(path), exist_ok=True) # Stellt sicher, dass das Verzeichnis existiert
        torch.save(self.qnetwork_local.state_dict(), path) # Speichert den state_dict (Parameter)
        print(f"DQN Agent gespeichert unter: {path}")

    def load(self, filename=None):
        """Lädt Gewichte für das lokale Q-Netzwerk (und synchronisiert das Zielnetzwerk)."""
        path = filename if filename else config.DQN_MODEL_PATH
        if os.path.exists(path):
            # Lädt Gewichte in das lokale Netzwerk
            self.qnetwork_local.load_state_dict(torch.load(path, map_location=device))
            # Synchronisiert das Zielnetzwerk mit den geladenen Gewichten
            self.qnetwork_target.load_state_dict(torch.load(path, map_location=device)) 
            self.qnetwork_local.eval()  # Nach dem Laden in den Evaluationsmodus setzen
            self.qnetwork_target.eval() # (wird im Training wieder auf .train() gesetzt)
            print(f"DQN Agent geladen von: {path}")
        else:
            print(f"FEHLER: Keine DQN-Agenten-Datei unter {path} gefunden.")
            raise FileNotFoundError(f"DQN Modelldatei nicht gefunden: {path}")