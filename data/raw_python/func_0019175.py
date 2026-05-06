def networkproperties(self):
        """Print out some properties of the network defined by the |Node| and
        |Element| objects currently handled by the |HydPy| object."""
        print('Number of nodes: %d' % len(self.nodes))
        print('Number of elements: %d' % len(self.elements))
        print('Number of end nodes: %d' % len(self.endnodes))
        print('Number of distinct networks: %d' % len(self.numberofnetworks))
        print('Applied node variables: %s' % ', '.join(self.variables))