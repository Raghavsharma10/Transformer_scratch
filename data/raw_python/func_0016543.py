def reset(self):
        """Reset the scenario, starting it over for a new run.

        Usage:
            if not scenario.more():
                scenario.reset()

        Arguments: None
        Return: None
        """
        self.remaining_cycles = self.initial_training_cycles
        self.needle_index = random.randrange(self.input_size)