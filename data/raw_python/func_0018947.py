def copy(self, name: str) -> 'Selection':
        """Return a new |Selection| object with the given name and copies
        of the handles |Nodes| and |Elements| objects based on method
        |Devices.copy|."""
        return type(self)(name, copy.copy(self.nodes), copy.copy(self.elements))