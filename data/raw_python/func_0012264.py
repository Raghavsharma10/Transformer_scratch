def add_before(self):
        """Returns a builder inserting a new block before the current block"""
        idx = self._container.structure.index(self)
        return BlockBuilder(self._container, idx)