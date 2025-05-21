# lunar_lander_agents/agents/ppo_agent.py
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
# from collections import deque # deque wird hier nicht direkt für den Hauptspeicher verwendet

# Importiert die globale Konfigurationsdatei
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

# PPO verwendet typischerweise separate Netzwerke für Actor und Critic,
# oder ein gemeinsames Netzwerk mit separaten "Köpfen" (Ausgabeschichten).
# Hier sind sie als separate Klassen implementiert.

class ActorPPO(nn.Module): # Erbt von PyTorch's nn.Module
    """Definiert das Actor-Netzwerk für PPO, das die Policy approximiert."""
    def __init__(self, state_size, action_size, seed, 
                 fc1_units=config.PPO_ACTOR_FC1, # Nutzt Architekturparameter aus config.py
                 fc2_units=config.PPO_ACTOR_FC2):
        super(ActorPPO, self).__init__()
        self.seed = torch.manual_seed(seed) # Setzt den PyTorch-Seed für diese Netzwerkinstanz
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_actor = nn.Linear(fc2_units, action_size) # Ausgabeschicht gibt Logits für Aktionen

    def forward(self, state):
        """Definiert den Forward-Pass des Actor-Netzwerks."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc_actor(x) # Gibt rohe Aktions-Logits zurück
        return action_logits

class CriticPPO(nn.Module): # Erbt von PyTorch's nn.Module
    """Definiert das Critic-Netzwerk für PPO, das die Value-Funktion V(s) approximiert."""
    def __init__(self, state_size, seed, 
                 fc1_units=config.PPO_CRITIC_FC1, # Nutzt Architekturparameter aus config.py
                 fc2_units=config.PPO_CRITIC_FC2):
        super(CriticPPO, self).__init__()
        self.seed = torch.manual_seed(seed) # Setzt den PyTorch-Seed
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_critic = nn.Linear(fc2_units, 1) # Ausgabeschicht gibt einen einzelnen Wert (V(s)) aus

    def forward(self, state):
        """Definiert den Forward-Pass des Critic-Netzwerks."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc_critic(x) # Gibt den geschätzten Zustandswert zurück
        return value

class PPOAgent(BaseAgent): # Erbt von der BaseAgent-Klasse
    """
    Implementiert den Proximal Policy Optimization (PPO) Algorithmus.
    PPO ist ein On-Policy Actor-Critic Algorithmus, der für seine Stabilität und Effizienz bekannt ist.
    Er sammelt Erfahrungen über einen bestimmten Horizont und führt dann mehrere Optimierungsepochen durch.
    """
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 seed=0, actor_lr=config.PPO_ACTOR_LR, critic_lr=config.PPO_CRITIC_LR, # Lernraten aus config
                 gamma=config.PPO_GAMMA, ppo_epochs=config.PPO_EPOCHS, 
                 ppo_clip=config.PPO_CLIP_EPSILON, batch_size=config.PPO_BATCH_SIZE,
                 gae_lambda=config.PPO_GAE_LAMBDA, entropy_coeff=config.PPO_ENTROPY_COEFF, 
                 value_loss_coeff=config.PPO_VALUE_LOSS_COEFF,
                 actor_fc1=config.PPO_ACTOR_FC1, actor_fc2=config.PPO_ACTOR_FC2,
                 critic_fc1=config.PPO_CRITIC_FC1, critic_fc2=config.PPO_CRITIC_FC2,
                 update_horizon=config.PPO_UPDATE_HORIZON): # Anzahl Schritte pro Datensammlung
        super().__init__(observation_space, action_space) # Ruft Konstruktor der Basisklasse auf

        # Speichert wichtige Konfigurations- und Hyperparameter
        self.state_size = observation_space.shape[0]
        self.action_size = action_space.n
        self._agent_seed = seed
        self.gamma = gamma                # Diskontierungsfaktor
        self.gae_lambda = gae_lambda      # Lambda-Parameter für Generalized Advantage Estimation (GAE)
        self.ppo_epochs = ppo_epochs      # Anzahl der Optimierungsepochen über die gesammelten Daten
        self.ppo_clip = ppo_clip          # Clipping-Parameter für die PPO Surrogate Objective Function
        self.entropy_coeff = entropy_coeff # Koeffizient für den Entropie-Bonus
        self.value_loss_coeff = value_loss_coeff # Gewicht des Critic-Verlusts
        self.batch_size = batch_size      # Minibatch-Größe für SGD innerhalb der PPO-Epochen
        self.update_horizon = update_horizon # Anzahl der Schritte, die gesammelt werden, bevor ein Update erfolgt

        # Seeds für Reproduzierbarkeit setzen
        random.seed(self._agent_seed)
        torch.manual_seed(self._agent_seed)
        np.random.seed(self._agent_seed) # Wichtig für das Mischen der Indizes bei Minibatch-SGD
        if config.DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(self._agent_seed)

        # Actor- und Critic-Netzwerke initialisieren und auf das konfigurierte Gerät verschieben
        self.actor = ActorPPO(self.state_size, self.action_size, self._agent_seed, actor_fc1, actor_fc2).to(config.DEVICE)
        self.critic = CriticPPO(self.state_size, self._agent_seed, critic_fc1, critic_fc2).to(config.DEVICE)
        
        # Separate Adam-Optimierer für Actor und Critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Speicherlisten für die gesammelten Trajektoriendaten eines Update-Horizonts
        self.memory_states = []
        self.memory_actions = []
        self.memory_log_probs = [] # Log-Wahrscheinlichkeiten der Aktionen unter der *alten* Policy
        self.memory_rewards = []
        self.memory_dones = []
        self.memory_values = []    # Vom Critic geschätzte Zustandswerte V(s_t) zum Zeitpunkt der Aktion

        print(f"PPO Agent initialisiert. Gerät: {config.DEVICE}")

    def select_action(self, state, store_in_memory=True):
        """Wählt eine Aktion basierend auf der aktuellen Actor-Policy und speichert relevante Daten."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(config.DEVICE)
        
        # Netzwerke in den Evaluationsmodus für die Inferenz setzen (kein Dropout etc.)
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad(): # Keine Gradientenberechnung für diesen Teil
            action_logits = self.actor(state_tensor) # Aktions-Logits vom Actor
            value = self.critic(state_tensor)        # Zustandswert V(s_t) vom Critic
        # Netzwerke zurück in den Trainingsmodus für nachfolgende Lernschritte
        self.actor.train()
        self.critic.train()

        action_probs = F.softmax(action_logits, dim=-1) # Logits in Wahrscheinlichkeiten
        m = Categorical(action_probs)                   # Kategoriale Verteilung
        action = m.sample()                             # Aktion sampeln
        log_prob = m.log_prob(action)                   # Log-Wahrscheinlichkeit der gesampelten Aktion

        # Wenn für das Training (Datensammlung), dann relevante Daten im Speicher ablegen
        if store_in_memory:
            self.memory_states.append(state) # Zustand als NumPy-Array speichern
            self.memory_actions.append(action.item()) # Aktion als Python-Integer
            self.memory_log_probs.append(log_prob.item()) # Log-Wahrscheinlichkeit als Python-Float
            self.memory_values.append(value.item())       # Geschätzten Wert V(s_t) als Python-Float

        return action.item() # Aktion als Python-Integer zurückgeben

    def store_transition_result(self, reward, done):
        """Speichert Belohnung und Beendigungsstatus nach Ausführung eines Schritts."""
        # Stellt sicher, dass nur gespeichert wird, wenn zuvor auch eine Aktion (und deren Daten) gespeichert wurde
        if len(self.memory_rewards) < len(self.memory_actions): 
            self.memory_rewards.append(reward)
            self.memory_dones.append(done)

    def _calculate_advantages_gae(self, next_value_tensor):
        """Berechnet Advantages und Returns mittels Generalized Advantage Estimation (GAE)."""
        advantages = [] # Liste für die berechneten GAE-Advantages
        gae = 0         # Initialisiert den GAE-Term
        
        # Konvertiert die Python-Listen aus dem Speicher in PyTorch-Tensoren
        rewards_t = torch.tensor(self.memory_rewards, dtype=torch.float32).to(config.DEVICE)
        dones_t = torch.tensor(self.memory_dones, dtype=torch.float32).to(config.DEVICE)
        values_t = torch.tensor(self.memory_values, dtype=torch.float32).to(config.DEVICE) # Dies sind V(s_t)

        # Iteriert rückwärts durch die gesammelten Erfahrungen des Horizonts
        for i in reversed(range(len(rewards_t))):
            # Bestimmt den Wert des Folgezustands V(s_{t+1})
            # Wenn i der letzte Schritt im Horizont ist, ist v_next der übergebene next_value_tensor (Wert von S_T)
            # Ansonsten ist es der im Speicher abgelegte Wert values_t[i+1] = V(s_{i+1})
            v_next = values_t[i+1] if i + 1 < len(values_t) else next_value_tensor.squeeze()
            
            # TD-Error (delta_t) = R_t + gamma * V(s_{t+1}) * (1-done_t) - V(s_t)
            delta = rewards_t[i] + self.gamma * v_next * (1 - dones_t[i]) - values_t[i]
            # GAE-Formel: A_GAE(t) = delta_t + gamma * lambda * (1-done_t) * A_GAE(t+1)
            gae = delta + self.gamma * self.gae_lambda * (1 - dones_t[i]) * gae
            advantages.insert(0, gae) # Advantage am Anfang der Liste einfügen
        
        advantages_tensor = torch.stack(advantages) if advantages else torch.empty(0, device=config.DEVICE)
        # Returns (Ziele für den Critic) = Advantages + V(s_t)
        returns_tensor = advantages_tensor + values_t 
        
        # Normalisierung der Advantages (üblich zur Stabilisierung)
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8) # 1e-8 zur Vermeidung von Division durch Null
        
        return advantages_tensor, returns_tensor


    def learn_from_memory(self, last_observation_if_not_done):
        """Führt die PPO-Optimierungsschleife über die gesammelten Daten des Horizonts durch."""
        # Nicht genug Daten für ein Update gesammelt
        if len(self.memory_actions) < self.update_horizon and len(self.memory_actions) > 0 : # len > 0 hinzugefügt, um Fehler bei leerem Speicher zu vermeiden
             # Diese Bedingung sollte eigentlich durch den Aufruf in train.py abgedeckt sein,
             # aber als zusätzliche Sicherheit.
            return

        # Schätzt den Wert des Zustands nach dem letzten gesammelten Schritt im Horizont,
        # falls die Episode dort nicht beendet war (Bootstrap-Wert).
        next_value_tensor = torch.zeros(1, 1, device=config.DEVICE) # Standardwert, falls letzter Schritt terminal war
        if not self.memory_dones[-1] and last_observation_if_not_done is not None:
            with torch.no_grad():
                state_t = torch.from_numpy(last_observation_if_not_done).float().unsqueeze(0).to(config.DEVICE)
                next_value_tensor = self.critic(state_t) # V(S_T)

        # Advantages und Returns (Ziele für Critic) mit GAE berechnen
        advantages, returns = self._calculate_advantages_gae(next_value_tensor)
        
        # Übrige Speicherlisten in Tensoren umwandeln
        old_states_t = torch.from_numpy(np.array(self.memory_states)).float().to(config.DEVICE)
        old_actions_t = torch.tensor(self.memory_actions, dtype=torch.long).to(config.DEVICE)
        old_log_probs_t = torch.tensor(self.memory_log_probs, dtype=torch.float32).to(config.DEVICE) # Log-Probs der alten Policy

        # --- PPO Optimierungs-Schleife (mehrere Epochen über die gleichen Daten) ---
        num_samples = len(old_actions_t)
        indices = np.arange(num_samples) # Indizes für das Mischen

        for _ in range(self.ppo_epochs): # Iteriere PPO_EPOCHS Mal über die Daten
            np.random.shuffle(indices) # Mische die Indizes für Minibatch SGD
            for start in range(0, num_samples, self.batch_size): # Iteriere über Minibatches
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Minibatch extrahieren
                mb_states = old_states_t[batch_indices]
                mb_actions = old_actions_t[batch_indices]
                mb_old_log_probs = old_log_probs_t[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices] # Ziele für den Critic

                # Werte mit der *aktuellen* Policy und dem *aktuellen* Critic evaluieren
                action_logits_new = self.actor(mb_states) # Neue Aktions-Logits
                current_values_new = self.critic(mb_states).squeeze() # Neue Zustandswerte V(s)

                m_new = Categorical(F.softmax(action_logits_new, dim=-1))
                new_log_probs = m_new.log_prob(mb_actions) # Neue Log-Wahrscheinlichkeiten der alten Aktionen
                entropy = m_new.entropy().mean()           # Entropie der neuen Policy-Verteilung

                # Verhältnis (Ratio) r_t = pi_new(a|s) / pi_old(a|s) = exp(log_pi_new - log_pi_old)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # === PPO Clipped Surrogate Objective (Actor-Verlust) ===
                # Ungeclippter Term: ratio * Advantage
                surr1 = ratio * mb_advantages
                # Geclippter Term: clamp(ratio, 1-clip, 1+clip) * Advantage
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_advantages
                # Nimm das Minimum der beiden Terme und negiere es (da Optimierer minimieren)
                actor_loss = -torch.min(surr1, surr2).mean()

                # === Critic-Verlust (Value-Loss) ===
                # MSE zwischen den neuen Wertschätzungen und den berechneten Returns (Zielen)
                critic_loss = F.mse_loss(current_values_new, mb_returns)
                
                # === Gesamtverlust (kombiniert für separate Updates oder gemeinsames Netzwerk) ===
                # Hier werden separate Optimierer verwendet.
                # loss = actor_loss + self.value_loss_coeff * critic_loss - self.entropy_coeff * entropy # Nicht direkt für separate Optimizer genutzt

                # --- Actor Update ---
                self.actor_optimizer.zero_grad()
                # actor_loss (plus ggf. Entropie-Term) für Actor-Update verwenden
                (actor_loss - self.entropy_coeff * entropy).backward() # Entropie hilft Actor bei Exploration
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5) # Optional: Gradient Clipping
                self.actor_optimizer.step()

                # --- Critic Update ---
                self.critic_optimizer.zero_grad()
                critic_loss.backward() # Nur Critic-Verlust für Critic-Update
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5) # Optional: Gradient Clipping
                self.critic_optimizer.step()
        
        # Speicher nach den Update-Epochen leeren für die nächste Datensammlungsphase
        self.clear_memory()

    def learn(self, observation, action, reward, next_observation, terminated, truncated):
        """
        Für PPO dient diese Methode primär dem Speichern der Erfahrung.
        Das eigentliche Lernen geschieht in `learn_from_memory()`, nachdem genügend Daten gesammelt wurden.
        """
        # Die Aktion wurde bereits in `select_action` gewählt und relevante Daten (Zustand, Aktion, log_prob, Wert) gespeichert.
        # Hier wird nur das Ergebnis des Schritts (Belohnung, Beendigung) gespeichert.
        self.store_transition_result(reward, terminated or truncated)
        # Der Trainings-Loop in `train.py` prüft dann, wann `learn_from_memory()` aufgerufen werden soll.

    def clear_memory(self):
        """Leert alle Speicherlisten für die nächste Datensammlungsrunde."""
        self.memory_states = []
        self.memory_actions = []
        self.memory_log_probs = []
        self.memory_rewards = []
        self.memory_dones = []
        self.memory_values = []

    def reset(self):
        """
        PPO ist on-policy. Ein explizites Reset des Agenten-Zustands pro Episode ist meist nicht nötig,
        da der Speicher in `learn_from_memory` geleert wird.
        """
        pass

    def save(self, filename_actor, filename_critic): # PPO speichert zwei separate Modelle
        """Speichert die Gewichte des Actor- und Critic-Netzwerks."""
        # Sicherstellen, dass die Zielverzeichnisse existieren
        os.makedirs(os.path.dirname(filename_actor), exist_ok=True)
        os.makedirs(os.path.dirname(filename_critic), exist_ok=True)
        torch.save(self.actor.state_dict(), filename_actor)
        torch.save(self.critic.state_dict(), filename_critic)
        print(f"PPO Actor gespeichert unter {filename_actor}, Critic gespeichert unter {filename_critic}")

    def load(self, filename_actor, filename_critic):
        """Lädt die Gewichte für Actor und Critic."""
        loaded_actor = False
        loaded_critic = False
        if os.path.exists(filename_actor):
            self.actor.load_state_dict(torch.load(filename_actor, map_location=config.DEVICE))
            self.actor.eval() # In den Evaluationsmodus setzen
            print(f"PPO Actor geladen von {filename_actor}")
            loaded_actor = True
        else:
            print(f"FEHLER: PPO Actor-Datei nicht unter {filename_actor} gefunden.")

        if os.path.exists(filename_critic):
            self.critic.load_state_dict(torch.load(filename_critic, map_location=config.DEVICE))
            self.critic.eval() # In den Evaluationsmodus setzen
            print(f"PPO Critic geladen von {filename_critic}")
            loaded_critic = True
        else:
            print(f"FEHLER: PPO Critic-Datei nicht unter {filename_critic} gefunden.")
        
        # Fehler auslösen, wenn nicht beide Modelle erfolgreich geladen werden konnten
        if not (loaded_actor and loaded_critic):
            raise FileNotFoundError("PPO Modelldateien (Actor oder Critic oder beide) nicht gefunden.")