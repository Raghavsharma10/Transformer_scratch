def add_after(self):
        """Returns a builder inserting a new block after the current block"""
        idx = self._container.structure.index(self)
        return BlockBuilder(self._container, idx+1)