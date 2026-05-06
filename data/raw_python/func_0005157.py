def set_level(self, level):
        """ Sets :attr:loglevel to @level

            @level: #str one or several :attr:levels
        """
        if not level:
            return None
        self.levelmap = set()
        for char in level:
            self.levelmap = self.levelmap.union(self.levels[char])
        self.loglevel = level
        return self.loglevel