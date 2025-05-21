# lunar_lander_agents/agents/random_agent.py
from .base_agent import BaseAgent # Importiert die abstrakte Basisklasse für Agenten
import gymnasium as gym # Importiert die Gymnasium-Bibliothek

class RandomAgent(BaseAgent): # Definiert den RandomAgent, der von BaseAgent erbt
    """
    Ein einfacher Agent, der zufällige Aktionen auswählt.
    Dient als Baseline für den Vergleich mit lernfähigen Agenten.
    """
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        """
        Initialisiert den RandomAgent.

        Args:
            observation_space (gym.spaces.Space): Der Beobachtungsraum der Umgebung. (Wird hier nicht aktiv genutzt)
            action_space (gym.spaces.Space): Der Aktionsraum der Umgebung. (Wichtig für die Aktionsauswahl)
        """
        # Ruft den Konstruktor der Basisklasse auf, um observation_space und action_space zu speichern.
        super().__init__(observation_space, action_space)
        print(f"RandomAgent initialisiert. Aktionsraum: {self.action_space}")

    def select_action(self, observation):
        """
        Wählt eine zufällige Aktion aus dem Aktionsraum der Umgebung aus.
        Ignoriert die gegebene Beobachtung (observation).

        Args:
            observation: Die aktuelle Beobachtung der Umgebung (wird nicht verwendet).

        Returns:
            Eine zufällig ausgewählte Aktion.
        """
        # Nutzt die sample()-Methode des Aktionsraums, um eine gültige zufällige Aktion zu erhalten.
        return self.action_space.sample()

    # Die Methoden `learn` und `reset` der BaseAgent-Klasse sind für diesen Agenten ausreichend,
    # da er weder lernt noch einen episodenspezifischen Zustand hat, der zurückgesetzt werden müsste.
    # Daher werden sie hier nicht überschrieben.