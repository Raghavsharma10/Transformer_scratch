def unblock_all(self):
        """ Unblock all emitters in this group.
        """
        self.unblock()
        for em in self._emitters.values():
            em.unblock()