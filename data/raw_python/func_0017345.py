def _match_transition(self, transition):
        """Checks whether a given Transition matches self.names."""
        return (self.names == '*'
                or transition in self.names
                or transition.name in self.names)