# lunar_lander_agents/agents/base_agent.py

# Importiert ABC (Abstract Base Classes) und abstractmethod aus dem Modul abc.
# Dies wird verwendet, um eine abstrakte Basisklasse zu definieren, die nicht direkt instanziiert werden kann
# und abstrakte Methoden erzwingen kann, die von Unterklassen implementiert werden müssen.
from abc import ABC, abstractmethod
import gymnasium as gym # Importiert die Gymnasium-Bibliothek für Umgebungsinteraktionen

class BaseAgent(ABC): # Definiert die Klasse BaseAgent, die von ABC erbt
    """
    Abstrakte Basisklasse für einen Agenten.
    Diese Klasse definiert die grundlegende Schnittstelle, die alle spezifischen Agentenimplementierungen
    (z.B. DQN, PPO, RandomAgent) erfüllen müssen. Sie stellt sicher, dass alle Agenten
    konsistent in den Trainings-, Test- und Evaluierungsskripten verwendet werden können.
    """
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        """
        Initialisiert den Agenten. Wird von allen Unterklassen aufgerufen.

        Args:
            observation_space (gym.spaces.Space): Der Beobachtungsraum der Umgebung.
                Enthält Informationen über die Struktur und Grenzen der Beobachtungen
                (z.B. Dimensionen, Datentyp, Min/Max-Werte).
            action_space (gym.spaces.Space): Der Aktionsraum der Umgebung.
                Definiert die möglichen Aktionen, die der Agent ausführen kann
                (z.B. Anzahl diskreter Aktionen oder Grenzen für kontinuierliche Aktionen).
        """
        # Speichert die Beobachtungs- und Aktionsräume als Instanzvariablen,
        # damit der Agent darauf zugreifen kann, z.B. um die Größe von Eingabe-
        # oder Ausgabeschichten von neuronalen Netzen zu bestimmen.
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod # Markiert die Methode als abstrakt
    def select_action(self, observation):
        """
        Wählt eine Aktion basierend auf der aktuellen Beobachtung aus.
        Diese Methode MUSS von jeder konkreten Unterklasse implementiert werden,
        da die Logik zur Aktionsauswahl spezifisch für jeden Agententyp ist.

        Args:
            observation: Die aktuelle Beobachtung (Zustand) von der Umgebung.
                         Typischerweise ein NumPy-Array oder ein ähnliches Datenformat.

        Returns:
            Die vom Agenten gewählte Aktion. Der Typ der Aktion hängt vom
            Aktionsraum der Umgebung ab (z.B. ein Integer für diskrete Aktionen).
        """
        pass # Abstrakte Methoden haben keine Implementierung in der Basisklasse

    def learn(self, observation, action, reward, next_observation, terminated, truncated):
        """
        Optionale Methode für Agenten, die aus Erfahrung lernen (z.B. Reinforcement Learning Agenten).
        Diese Methode wird typischerweise nach jedem Schritt des Agenten in der Umgebung aufgerufen.
        Die Standardimplementierung tut nichts; lernfähige Agenten überschreiben diese Methode,
        um ihre Lernalgorithmen (z.B. Update von Q-Werten, Policy-Gradienten-Updates) auszuführen.

        Args:
            observation: Die Beobachtung (Zustand S_t) vor der Ausführung der Aktion.
            action: Die vom Agenten ausgeführte Aktion (A_t).
            reward: Die Belohnung (R_{t+1}), die nach Ausführung der Aktion erhalten wurde.
            next_observation: Die Beobachtung (Zustand S_{t+1}) nach Ausführung der Aktion.
            terminated (bool): Ein Flag, das anzeigt, ob die Episode aufgrund eines
                               terminalen Zustands (z.B. Ziel erreicht, abgestürzt) beendet wurde.
            truncated (bool): Ein Flag, das anzeigt, ob die Episode aufgrund einer externen
                                Bedingung (z.B. Zeitlimit erreicht) abgebrochen wurde, ohne dass
                                ein terminaler Zustand der Aufgabe erreicht wurde.
        """
        # Agenten, die nicht lernen (z.B. RandomAgent) oder deren Lernlogik
        # an anderer Stelle stattfindet (z.B. REINFORCE am Ende der Episode, GA/ES über Generationen),
        # müssen diese Methode nicht implementieren oder können die Standardimplementierung beibehalten.
        pass

    def reset(self):
        """
        Optionale Methode zum Zurücksetzen des internen Zustands des Agenten zu Beginn einer neuen Episode.
        Nützlich für Agenten, die episodenspezifische Informationen speichern (z.B. REINFORCE)
        oder Parameter haben, die pro Episode angepasst werden (z.B. Epsilon-Decay bei DQN).
        Die Standardimplementierung tut nichts.
        """
        pass

    # Zukünftige Erweiterungen könnten hier Standardmethoden für das Speichern und Laden
    # von Modellen definieren, die von Unterklassen überschrieben werden können,
    # z.B. @abstractmethod def save(self, filepath) und @abstractmethod def load(self, filepath).
    # Aktuell wird dies direkt in den Unterklassen gehandhabt.