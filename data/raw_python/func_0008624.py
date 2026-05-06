def copy(self):
        """Return a new instance with the same attributes."""
        return self.__class__([b.copy() for b in self.blocks],
                tuple(self.pos) if self.pos else None)