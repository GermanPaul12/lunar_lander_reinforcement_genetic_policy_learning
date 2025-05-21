# lunar_lander_agents/agents/__init__.py

# Importiert die Basisklasse für alle Agenten.
# Jede spezifische Agentenimplementierung erbt von dieser Klasse.
from .base_agent import BaseAgent

# Importiert die Implementierung des Zufallsagenten.
# Dieser Agent wählt Aktionen zufällig aus und dient als Baseline.
from .random_agent import RandomAgent

# Importiert den Deep Q-Network (DQN) Agenten.
# Beinhaltet die DQNAgent-Klasse und potenziell zugehörige Netzwerkklassen (QNetwork),
# obwohl diese oft innerhalb der dqn_agent.py-Datei gekapselt sind.
from .dqn_agent import DQNAgent

# Importiert den Genetischen Algorithmus (GA) Agenten und zugehörige Klassen.
# GeneticAgent: Der Agenten-Wrapper, der die beste gefundene Policy des GA verwendet.
# GeneticAlgorithmController: Verwaltet den evolutionären Prozess (Population, Selektion, Crossover, Mutation).
# PolicyNetwork: Die Netzwerkarchitektur, die von den Individuen des GA verwendet wird.
from .genetic_agent import GeneticAgent, GeneticAlgorithmController, PolicyNetwork as GAPolicyNetwork # Umbenannt für Klarheit

# Importiert den REINFORCE (Monte Carlo Policy Gradient) Agenten.
# REINFORCEAgent: Die Hauptklasse für den Algorithmus.
# PolicyNetworkREINFORCE: Die spezifische Policy-Netzwerkarchitektur für REINFORCE.
from .reinforce_agent import REINFORCEAgent, PolicyNetworkREINFORCE

# Importiert den Advantage Actor-Critic (A2C) Agenten.
# A2CAgent: Die Hauptklasse, die sowohl den Actor als auch den Critic verwaltet.
# Zugehörige Netzwerkklassen (z.B. ActorCriticNetwork) sind typischerweise innerhalb von a2c_agent.py definiert.
from .a2c_agent import A2CAgent

# Importiert den Proximal Policy Optimization (PPO) Agenten.
# PPOAgent: Die Hauptklasse für diesen fortgeschrittenen Policy-Gradienten-Algorithmus.
# Zugehörige Netzwerkklassen (z.B. ActorPPO, CriticPPO) sind typischerweise innerhalb von ppo_agent.py definiert.
from .ppo_agent import PPOAgent 

# Importiert den Evolutionäre Strategien (ES) Agenten.
# ESAgent: Agiert als Controller für den ES-Optimierungsprozess und stellt die beste gefundene Policy bereit.
# PolicyNetworkES: Die spezifische Policy-Netzwerkarchitektur, deren Parameter durch ES optimiert werden.
from .es_agent import ESAgent, PolicyNetworkES


# AGENT_REGISTRY: Ein Dictionary, das eine einfache Möglichkeit bietet,
# auf die Hauptklasse eines Agenten anhand eines String-Schlüssels zuzugreifen.
# Dies wird in den Skripten (train.py, test.py, evaluate.py) verwendet,
# um Agenten dynamisch basierend auf der Konfiguration zu instanziieren.
AGENT_REGISTRY = {
    "random": RandomAgent,          # Schlüssel "random" verweist auf die RandomAgent Klasse
    "dqn": DQNAgent,                # Schlüssel "dqn" verweist auf die DQNAgent Klasse
    "genetic": GeneticAgent,        # Schlüssel "genetic" verweist auf den GeneticAgent Wrapper
    "reinforce": REINFORCEAgent,    # Schlüssel "reinforce" verweist auf die REINFORCEAgent Klasse
    "a2c": A2CAgent,                # Schlüssel "a2c" verweist auf die A2CAgent Klasse
    "ppo": PPOAgent,                # Schlüssel "ppo" verweist auf die PPOAgent Klasse
    "es": ESAgent,                  # Schlüssel "es" verweist auf die ESAgent Klasse (die als Agent agiert)
}

# Hinweis: Controller-Klassen (wie GeneticAlgorithmController) oder spezifische Netzwerkklassen,
# die nicht direkt als "der Agent" für die Interaktion mit der Umgebung dienen,
# werden typischerweise nicht in AGENT_REGISTRY aufgenommen.
# Sie können bei Bedarf direkt aus ihrem jeweiligen Modul importiert werden
# (z.B. `from agents.genetic_agent import GeneticAlgorithmController`).
# Die PolicyNetwork-Klassen wurden hier importiert, um sie potenziell außerhalb
# verfügbar zu machen, falls sie z.B. für das direkte Laden von Gewichten ohne
# die vollständige Agenten-Wrapper-Klasse benötigt werden (obwohl dies seltener der Fall ist).
# Umbenennung von `PolicyNetwork` zu `GAPolicyNetwork` beim Import dient der Vermeidung von Namenskonflikten,
# falls andere Agenten ebenfalls eine Klasse namens `PolicyNetwork` definieren.