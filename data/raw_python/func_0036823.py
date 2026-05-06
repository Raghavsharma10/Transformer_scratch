def available_string(self, episode):
        """Return a string of available episodes."""
        available = [ep for ep in self if ep > episode]
        string = ','.join(str(ep) for ep in available[:self.EPISODES_TO_SHOW])
        if len(available) > self.EPISODES_TO_SHOW:
            string += '...'
        return string