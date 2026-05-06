def get_new_edges(self, level):
        """Get new edges from the pattern graph for the graph search algorithm

           The level argument denotes the distance of the new edges from the
           starting vertex in the pattern graph.
        """
        return (
            self.level_edges.get(level, []),
            self.level_constraints.get(level, [])
        )