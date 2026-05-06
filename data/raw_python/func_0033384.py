def add_play(self, choice, count=1):
        """Increments the play count for a given experiment choice"""
        self.redis.hincrby(EXPERIMENT_REDIS_KEY_TEMPLATE % self.name, "%s:plays" % choice, count)
        self._choices = None