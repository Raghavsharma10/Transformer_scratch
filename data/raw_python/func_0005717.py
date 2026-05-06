def spawn_generator(self, g):
        """
        Return a fresh spawn of g unless g is already
        contained in this SpawnMapping, in which case
        return the previously spawned generator.
        """
        try:
            return self.mapping[g]
        except KeyError:
            return g._spawn(self)