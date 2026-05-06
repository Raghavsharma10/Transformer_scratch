def get_actor_by_ain(self, ain):
        """
        Return a actor identified by it's ain or return None
        """
        for actor in self.get_actors():
            if actor.actor_id == ain:
                return actor