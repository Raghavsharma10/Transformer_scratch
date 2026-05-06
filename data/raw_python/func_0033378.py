def refresh(self):
        """Re-pulls the data from redis"""
        pipe = self.redis.pipeline()
        pipe.hget(EXPERIMENT_REDIS_KEY_TEMPLATE % self.name, "metadata")
        pipe.hget(EXPERIMENT_REDIS_KEY_TEMPLATE % self.name, "choices")
        pipe.hget(EXPERIMENT_REDIS_KEY_TEMPLATE % self.name, "default-choice")
        results = pipe.execute()

        if results[0] == None:
            raise ExperimentException(self.name, "Does not exist")

        self.metadata = parse_json(results[0])
        self.choice_names = parse_json(results[1]) if results[1] != None else []
        self.default_choice = escape.to_unicode(results[2])
        self._choices = None