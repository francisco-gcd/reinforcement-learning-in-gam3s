from abc import ABC
from abc import abstractmethod

class Algorithms(ABC):
    @abstractmethod
    def action(self, **arguments):
        pass

    @abstractmethod
    def learn(self, **arguments):
        pass

    @abstractmethod
    def next_episody(self):
        pass

    @abstractmethod
    def save(self, label):
        pass

    @abstractmethod
    def load(self, label):
        pass    