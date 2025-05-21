# lunar_lander_agents/agents/genetic_agent.py
import torch
import torch.nn as nn # PyTorch-Modul für neuronale Netze
import numpy as np
import random
from .base_agent import BaseAgent # Basisklasse für Agenten
import gymnasium as gym
import os
import copy # Für das Erstellen von tiefen Kopien von Objekten (wichtig für Individuen)

# Importiert die globale Konfigurationsdatei, um z.B. auf DEVICE zuzugreifen
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Fügt das Projekt-Stammverzeichnis zum Python-Pfad hinzu
import config

device = config.DEVICE # Verwendet das in config.py definierte Gerät (CPU oder CUDA)

# --- Policy-Netzwerk für einzelne Individuen des Genetischen Algorithmus ---
class PolicyNetwork(nn.Module): # Erbt von PyTorch's nn.Module
    """
    Definiert die Architektur des neuronalen Netzwerks, das als Policy für jedes
    Individuum im Genetischen Algorithmus dient.
    Nimmt einen Zustand als Eingabe und gibt Rohwerte (Scores/Logits) für jede Aktion aus.
    """
    def __init__(self, state_size, action_size, seed, fc1_units=config.GA_FC1_UNITS, fc2_units=config.GA_FC2_UNITS):
        """
        Initialisiert die Schichten des Netzwerks.

        Args:
            state_size (int): Dimension des Zustandsraums.
            action_size (int): Anzahl der möglichen Aktionen.
            seed (int): Seed für die Initialisierung der Gewichte zur Reproduzierbarkeit.
            fc1_units (int): Anzahl der Neuronen in der ersten versteckten Schicht.
            fc2_units (int): Anzahl der Neuronen in der zweiten versteckten Schicht.
        """
        super(PolicyNetwork, self).__init__()
        self.seed = torch.manual_seed(seed) # Setzt den PyTorch-Seed für diese Netzwerkinstanz
        # Definiert die linearen Schichten (fully connected layers)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size) # Ausgabeschicht gibt Scores für jede Aktion

    def forward(self, state):
        """Definiert den Forward-Pass des Netzwerks."""
        x = torch.relu(self.fc1(state)) # ReLU-Aktivierungsfunktion nach der ersten Schicht
        x = torch.relu(self.fc2(x)) # ReLU-Aktivierungsfunktion nach der zweiten Schicht
        return self.fc3(x) # Gibt die rohen Aktions-Scores zurück (kein Softmax hier)

    def get_weights(self):
        """Gibt alle Gewichte des Modells als einen flachen NumPy-Array zurück."""
        # Nützlich für genetische Operationen wie Crossover und Mutation.
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

    def set_weights(self, weights_flat):
        """Setzt die Gewichte des Modells aus einem flachen NumPy-Array."""
        offset = 0
        for param in self.parameters(): # Iteriert über alle Parameter (Gewichte und Biases) des Netzwerks
            shape = param.data.shape
            num_elements = param.data.numel() # Anzahl der Elemente im Parameter-Tensor
            # Kopiert den entsprechenden Teil des flachen Arrays in den Parameter-Tensor
            param.data.copy_(torch.from_numpy(weights_flat[offset:offset+num_elements]).view(shape).to(device))
            offset += num_elements

class GeneticAlgorithmController:
    """
    Verwaltet den evolutionären Prozess des Genetischen Algorithmus.
    Beinhaltet Population, Fitness-Evaluierung, Selektion, Crossover und Mutation.
    """
    def __init__(self, state_size, action_size, env_id=config.ENV_ID,
                 population_size=config.GA_POPULATION_SIZE,
                 n_generations=config.GA_N_GENERATIONS, # Wird vom train.py Skript übergeben
                 mutation_rate=config.GA_MUTATION_RATE,
                 mutation_strength=config.GA_MUTATION_STRENGTH,
                 crossover_rate=config.GA_CROSSOVER_RATE,
                 tournament_size=config.GA_TOURNAMENT_SIZE,
                 elitism_count=config.GA_ELITISM_COUNT,
                 eval_episodes_per_individual=config.GA_EVAL_EPISODES_PER_INDIVIDUAL,
                 max_steps_per_eval_episode=config.GA_MAX_STEPS_PER_GA_EVAL,
                 seed=0): # Seed wird übergeben (z.B. config.SEED)

        print(f"GA Controller initialisiert. Population: {population_size}, Generationen: {n_generations}, Seed: {seed}")
        # Speichert wichtige Konfigurationsparameter
        self.state_size = state_size
        self.action_size = action_size
        self.env_id = env_id # ID der Umgebung für interne Fitness-Evaluationen
        self.population_size = population_size
        # self.n_generations_config = n_generations # Speichert die konfigurierte Gesamtanzahl an Generationen
        
        # Hyperparameter für die genetischen Operatoren
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        
        # Parameter für die Fitness-Evaluation
        self.eval_episodes_per_individual = eval_episodes_per_individual
        self.max_steps_per_eval_episode = max_steps_per_eval_episode
        
        self.seed = seed # Seed für den Controller und die Initialpopulation
        random.seed(self.seed)
        np.random.seed(self.seed)
        # torch.manual_seed(self.seed) # PyTorch Seed wird pro Netzwerkinstanz gesetzt

        self.population = self._initialize_population() # Initialisiert die erste Population
        self.best_individual = None # Hält das bisher beste gefundene Individuum (PolicyNetwork)
        self.best_fitness = -float('inf') # Fitness des besten Individuums

    def _initialize_population(self):
        """Erstellt die initiale Population von Policy-Netzwerken."""
        population = []
        for i in range(self.population_size):
            # Jedes Netzwerk erhält einen eindeutigen Seed basierend auf dem Haupt-Seed
            # und seinem Index, um eine vielfältige Startpopulation zu gewährleisten.
            individual_seed = self.seed + i 
            policy_net = PolicyNetwork(self.state_size, self.action_size, seed=individual_seed,
                                       fc1_units=config.GA_FC1_UNITS, # Nutzt Architektur aus config
                                       fc2_units=config.GA_FC2_UNITS).to(device)
            population.append(policy_net)
        return population

    def _evaluate_fitness(self, individual_policy_net):
        """Evaluiert die Fitness eines einzelnen Individuums (Policy-Netzwerks)."""
        # Erstellt eine temporäre Umgebung für die Evaluation, um Interferenzen zu vermeiden.
        eval_env = gym.make(self.env_id) # Ohne Rendering für schnellere Evaluation
        total_rewards = []
        for _ in range(self.eval_episodes_per_individual): # Führt mehrere Episoden zur Mittelung durch
            observation, _ = eval_env.reset(seed=random.randint(0, 10000)) # Variabler Seed für Robustheit
            episode_reward = 0
            for _ in range(self.max_steps_per_eval_episode):
                state_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
                with torch.no_grad(): # Keine Gradientenberechnung während der Evaluation nötig
                    action_scores = individual_policy_net(state_tensor)
                action = torch.argmax(action_scores).item() # Wählt Aktion mit höchstem Score (deterministisch)
                observation, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            total_rewards.append(episode_reward)
        eval_env.close()
        return np.mean(total_rewards) # Durchschnittliche Belohnung als Fitness

    def _selection(self, fitnesses):
        """Wählt Eltern für die nächste Generation mittels Turnierselektion."""
        selected_parents = []
        # Wählt so viele Eltern, wie für die Erzeugung der neuen Generation (abzgl. Eliten) benötigt werden.
        for _ in range(self.population_size - self.elitism_count): 
            # Wählt zufällig 'tournament_size' Individuen für ein Turnier aus
            tournament_indices = random.sample(range(len(self.population)), self.tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            # Das Individuum mit der höchsten Fitness im Turnier gewinnt und wird als Elternteil ausgewählt
            winner_index_in_tournament = np.argmax(tournament_fitnesses)
            selected_parents.append(self.population[tournament_indices[winner_index_in_tournament]])
        return selected_parents

    def _crossover(self, parent1: PolicyNetwork, parent2: PolicyNetwork):
        """Führt Crossover zwischen zwei Elternteilen durch, um zwei Kinder zu erzeugen."""
        # Erstellt neue PolicyNetwork-Instanzen für die Kinder
        child1_net = PolicyNetwork(self.state_size, self.action_size, seed=random.randint(0,100000),
                                  fc1_units=config.GA_FC1_UNITS, fc2_units=config.GA_FC2_UNITS).to(device)
        child2_net = PolicyNetwork(self.state_size, self.action_size, seed=random.randint(0,100000),
                                  fc1_units=config.GA_FC1_UNITS, fc2_units=config.GA_FC2_UNITS).to(device)

        weights1 = parent1.get_weights() # Gewichte des ersten Elternteils
        weights2 = parent2.get_weights() # Gewichte des zweiten Elternteils

        if random.random() < self.crossover_rate: # Crossover mit einer bestimmten Wahrscheinlichkeit
            # Einfaches Durchschnitts-Crossover: Gewichte der Kinder sind der Mittelwert der Elterngewichte
            child1_weights = (weights1 + weights2) / 2.0
            child2_weights = (weights1 + weights2) / 2.0 # Hier könnte auch eine andere Logik stehen
            child1_net.set_weights(child1_weights)
            child2_net.set_weights(child2_weights)
        else: # Kein Crossover, Kinder sind Klone der Eltern
            child1_net.set_weights(weights1)
            child2_net.set_weights(weights2)
        return child1_net, child2_net

    def _mutate(self, individual_policy_net: PolicyNetwork):
        """Mutiert die Gewichte eines Individuums."""
        weights = individual_policy_net.get_weights()
        for i in range(len(weights)):
            if random.random() < self.mutation_rate: # Mutation mit einer bestimmten Wahrscheinlichkeit
                # Addiert Gaußsches Rauschen zu einem Gewicht
                noise = np.random.normal(0, self.mutation_strength)
                weights[i] += noise
        individual_policy_net.set_weights(weights)
        return individual_policy_net

    def evolve_population(self):
        """Führt einen kompletten Evolutionsschritt für eine Generation durch."""
        # 1. Fitness aller Individuen der aktuellen Population evaluieren
        fitnesses = [self._evaluate_fitness(ind) for ind in self.population]

        # 2. Bestes bisher gefundenes Individuum aktualisieren
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[current_best_idx]
            self.best_individual = copy.deepcopy(self.population[current_best_idx]) # Tiefe Kopie erstellen
            self.best_individual.to(device) # Sicherstellen, dass es auf dem richtigen Gerät ist
            print(f"    Neuer bester Fitnesswert: {self.best_fitness:.2f}")

        # 3. Neue Population erstellen
        new_population = []

        # Elitismus: Die besten 'elitism_count' Individuen direkt in die neue Population übernehmen
        sorted_indices = np.argsort(fitnesses)[::-1] # Indizes nach Fitness absteigend sortieren
        for i in range(self.elitism_count):
            elite_individual = copy.deepcopy(self.population[sorted_indices[i]])
            new_population.append(elite_individual.to(device))

        # 4. Restliche Population durch Selektion, Crossover und Mutation füllen
        parents = self._selection(fitnesses) # Eltern auswählen
        
        num_offspring_needed = self.population_size - self.elitism_count
        offspring_generated = 0
        parent_idx = 0
        while offspring_generated < num_offspring_needed:
            p1 = parents[parent_idx % len(parents)]
            # Nimmt den nächsten Parent, um sicherzustellen, dass p2 != p1, falls möglich
            p2 = parents[(parent_idx + 1) % len(parents)] 
            parent_idx +=1

            child1, child2 = self._crossover(p1, p2) # Crossover durchführen

            # Kinder mutieren und zur neuen Population hinzufügen
            new_population.append(self._mutate(child1))
            offspring_generated += 1
            if offspring_generated < num_offspring_needed: # Nur hinzufügen, wenn noch Platz ist
                new_population.append(self._mutate(child2))
                offspring_generated += 1

        self.population = new_population[:self.population_size] # Sicherstellen, dass die Populationsgröße korrekt ist
        return np.mean(fitnesses), np.max(fitnesses) # Gibt Durchschnitts- und Max-Fitness der alten Generation zurück

    def get_best_policy_network(self):
        """Gibt das beste bisher gefundene Policy-Netzwerk zurück."""
        # Fallback, falls `evolve_population` noch nicht aufgerufen wurde oder kein Bestes gefunden hat
        if self.best_individual is None and self.population:
            fitnesses = [self._evaluate_fitness(ind) for ind in self.population]
            best_idx = np.argmax(fitnesses)
            self.best_individual = copy.deepcopy(self.population[best_idx])
            self.best_individual.to(device)
            self.best_fitness = fitnesses[best_idx]
        return self.best_individual

    def save_best_individual(self, filename=None):
        """Speichert die Gewichte des besten Individuums."""
        path = filename if filename else config.GA_MODEL_PATH # Nutzt Pfad aus config.py falls keiner übergeben wird
        if self.best_individual:
            os.makedirs(os.path.dirname(path), exist_ok=True) # Stellt sicher, dass das Verzeichnis existiert
            torch.save(self.best_individual.state_dict(), path)
            print(f"Bestes GA-Individuum gespeichert unter: {path}")
        else:
            print("Kein bestes Individuum zum Speichern für GA vorhanden.")

    def load_best_individual(self, filename=None, state_size=None, action_size=None):
        """Lädt die Gewichte für das beste Individuum aus einer Datei."""
        path = filename if filename else config.GA_MODEL_PATH
        if os.path.exists(path):
            # state_size und action_size sind nötig, um die Netzwerkarchitektur korrekt zu instanziieren
            _state_size = state_size if state_size is not None else self.state_size
            _action_size = action_size if action_size is not None else self.action_size

            if _state_size is None or _action_size is None:
                print("Fehler: GA-Individuum kann nicht ohne state_size und action_size geladen werden.")
                return False

            # Erstellt eine neue Netzwerkinstanz mit der korrekten Architektur zum Laden der Gewichte
            loaded_net = PolicyNetwork(_state_size, _action_size, seed=0, # Seed hier irrelevant, da Gewichte geladen werden
                                       fc1_units=config.GA_FC1_UNITS, 
                                       fc2_units=config.GA_FC2_UNITS).to(device)
            loaded_net.load_state_dict(torch.load(path, map_location=device))
            loaded_net.eval() # In den Evaluationsmodus setzen
            self.best_individual = loaded_net
            # Optional könnte hier die Fitness des geladenen Individuums neu evaluiert werden
            print(f"Bestes GA-Individuum geladen von {path}. Fitness muss ggf. neu evaluiert werden.")
            return True
        else:
            print(f"Fehler: Keine GA-Modelldatei unter {path} gefunden.")
            return False

class GeneticAgent(BaseAgent): # Erbt von der BaseAgent-Klasse
    """
    Ein Agent, der die beste vom GeneticAlgorithmController gefundene Policy verwendet.
    Das eigentliche "Lernen" (Evolution) geschieht im Controller.
    """
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 policy_network: PolicyNetwork = None, seed=0): # Erwartet ein trainiertes PolicyNetwork
        super().__init__(observation_space, action_space)
        self.state_size = observation_space.shape[0]
        self.action_size = action_space.n # Nimmt diskreten Aktionsraum an
        self.policy_network = policy_network # Das zu verwendende Policy-Netzwerk
        self.seed = seed
        # Seeds für Reproduzierbarkeit setzen, falls der Agent selbst stochastische Elemente hätte
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.policy_network is None: # Fallback, falls kein Netzwerk übergeben wird
            print("GeneticAgent ohne Policy-Netzwerk initialisiert. Benötigt eines für Aktionen.")
            # Erstellt ein Dummy-Netzwerk, um Fehler zu vermeiden (dieses wird nicht gut performen)
            self.policy_network = PolicyNetwork(self.state_size, self.action_size, seed=self.seed,
                                               fc1_units=config.GA_FC1_UNITS, 
                                               fc2_units=config.GA_FC2_UNITS).to(device)
        else:
            self.policy_network.to(device) # Policy-Netzwerk auf das richtige Gerät verschieben
            self.policy_network.eval() # In den Evaluationsmodus setzen (kein Dropout etc.)

        print("GeneticAgent initialisiert.")

    def set_policy_network(self, policy_network: PolicyNetwork):
        """Ermöglicht das nachträgliche Setzen eines Policy-Netzwerks."""
        self.policy_network = policy_network
        if self.policy_network:
            self.policy_network.to(device)
            self.policy_network.eval()

    def select_action(self, observation):
        """Wählt eine Aktion basierend auf dem zugewiesenen Policy-Netzwerk."""
        if self.policy_network is None: # Sollte nicht passieren, wenn korrekt initialisiert
            return self.action_space.sample() # Fallback auf zufällige Aktion

        state_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
        with torch.no_grad(): # Keine Gradientenberechnung nötig
            action_scores = self.policy_network(state_tensor)
        return torch.argmax(action_scores).item() # Wählt Aktion mit höchstem Score

    def learn(self, observation, action, reward, next_observation, terminated, truncated):
        """
        Für dieses GA-Setup lernt der individuelle Agent (die Policy) nicht in einem einzelnen Schritt.
        Die Evolution geschieht im GeneticAlgorithmController über Generationen.
        """
        pass # Keine schrittweise Lernlogik hier

    def reset(self):
        """
        Setzt internen Zustand zurück, falls nötig. Für eine statische Policy meist nichts zu tun.
        """
        pass

    # Speichern/Laden bezieht sich hier auf die Policy dieses spezifischen Agenten-Wrappers,
    # üblicherweise würde man die beste Policy vom GAController speichern/laden.
    def save(self, filename="genetic_agent_policy.pth"):
        if self.policy_network:
            torch.save(self.policy_network.state_dict(), filename)
            print(f"Aktuelle Policy des GeneticAgent gespeichert unter: {filename}")
        else:
            print("GeneticAgent hat kein Policy-Netzwerk zum Speichern.")

    def load(self, filename="genetic_agent_policy.pth"):
        if os.path.exists(filename):
            # state_size und action_size sind nötig, um das Netzwerk zu rekonstruieren
            if self.policy_network is None:
                 self.policy_network = PolicyNetwork(self.state_size, self.action_size, seed=self.seed,
                                                    fc1_units=config.GA_FC1_UNITS,
                                                    fc2_units=config.GA_FC2_UNITS).to(device)
            self.policy_network.load_state_dict(torch.load(filename, map_location=device))
            self.policy_network.eval()
            print(f"Policy des GeneticAgent geladen von: {filename}")
        else:
            print(f"Fehler: Keine Policy-Datei unter {filename} für GeneticAgent gefunden.")