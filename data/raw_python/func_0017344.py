def _match_state(self, state):
        """Checks whether a given State matches self.names."""
        return (self.names == '*'
                or state in self.names
                or state.name in self.names)