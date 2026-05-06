def refresh(self):
        """Re-pulls the data from redis"""

        redis_key = EXPERIMENT_REDIS_KEY_TEMPLATE % self.experiment.name
        self.plays = int(self.experiment.redis.hget(redis_key, "%s:plays" % self.name) or 0)
        self.rewards = int(self.experiment.redis.hget(redis_key, "%s:rewards" % self.name) or 0)
        self.performance = float(self.rewards) / max(self.plays, 1)