def remove_choice(self, choice_name):
        """Adds a choice for the experiment"""

        self.choice_names.remove(choice_name)
        self.redis.hset(EXPERIMENT_REDIS_KEY_TEMPLATE % self.name, "choices", escape.json_encode(self.choice_names))
        self.refresh()