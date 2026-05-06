def edge_index(self):
        """A map to look up the index of a edge"""
        return dict((edge, index) for index, edge in enumerate(self.edges))