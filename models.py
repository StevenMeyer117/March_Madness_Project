import numpy as np


class Team:
    """
    Represents a basketball team with a name and strength rating.
    Provides a method to compute win probability against another team.
    """

    def __init__(self, name, strength):
        self.name = name
        self.strength = strength

    def win_probability(self, opponent):
        """
        Compute probability of this team beating another team
        using logistic function.
        """
        return self.strength / (self.strength + opponent.strength)