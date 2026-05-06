def copy(self, o=None):
        """Return a new instance, deep-copying all the attributes."""
        if o is None: o = self.__class__(self.project)
        o.scripts = [s.copy() for s in self.scripts]
        o.variables = dict((n, v.copy()) for (n, v) in self.variables.items())
        o.lists = dict((n, l.copy()) for (n, l) in self.lists.items())
        o.costumes = [c.copy() for c in self.costumes]
        o.sounds = [s.copy() for s in self.sounds]
        o.costume_index = self.costume_index
        o.volume = self.volume
        return o