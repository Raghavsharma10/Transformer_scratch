def add_choice(self, choice_name):
        """Adds a choice for the experiment"""

        if not ALLOWED_NAMES.match(choice_name):
            raise ExperimentException(self.name, "Illegal choice name: %s" % choice_name)

        if choice_name in self.choice_names:
            raise ExperimentException(self.name, "Choice already exists: %s" % choice_name)

        self.choice_names.append(choice_name)
        self.redis.hset(EXPERIMENT_REDIS_KEY_TEMPLATE % self.name, "choices", escape.json_encode(self.choice_names))
        self.refresh()