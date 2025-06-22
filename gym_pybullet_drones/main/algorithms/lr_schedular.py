'''https://github.com/TommyWoodley/TommyWoodleyMEngProject'''

class LinearLearningRateSchedule:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    """
    def __init__(self, initial_value: float, minimum_value: float = 0.00005):
        self.initial_value = initial_value
        self.minimum_value = minimum_value

    def __call__(self, progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: Remaining progress (1 to 0)
        :return: Current learning rate
        """
        return self.minimum_value + progress_remaining * self.initial_value

    def __repr__(self):
        return f"LinearLearningRateSchedule(initial_value={self.initial_value})"
