def clone(self, spawn_mapping=None):
        """
        Return an exact copy of this generator which behaves the same way
        (i.e., produces the same elements in the same order) and which is
        automatically reset whenever the original generator is reset.
        """
        c = self.spawn(spawn_mapping)
        self.register_clone(c)
        c.register_parent(self)
        return c