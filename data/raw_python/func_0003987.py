def get_new_edges(self, level):
        """Get new edges from the pattern graph for the graph search algorithm

           The level argument denotes the distance of the new edges from the
           starting vertex in the pattern graph.
        """
        if level == 0:
            edges0 = [(0, 1), (0, 2)]
        elif level >= (self.max_size-1)//2:
            edges0 = []
        else:
            l2 = level*2
            edges0 = [(l2-1, l2+1), (l2, l2+2)]
        return edges0, []