def block_all(self):
        """ Block all emitters in this group.
        """
        self.block()
        for em in self._emitters.values():
            em.block()