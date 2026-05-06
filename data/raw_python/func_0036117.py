def add(self, member, score):
        """Add the specified member to the sorted set, or update the score
        if it already exist."""
        return self.client.zadd(self.name, member, score)