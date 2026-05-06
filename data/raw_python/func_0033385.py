def compute_default_choice(self):
        """Computes and sets the default choice"""

        choices = self.choices

        if len(choices) == 0:
            return None

        high_choice = max(choices, key=lambda choice: choice.performance)
        self.redis.hset(EXPERIMENT_REDIS_KEY_TEMPLATE % self.name, "default-choice", high_choice.name)
        self.refresh()
        return high_choice