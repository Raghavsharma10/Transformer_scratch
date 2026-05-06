def connect_near(self, source, target, weight):
        # Near edges are added to self.near_graph, not self.graph, to avoid cycles
        """
        :type source: integer
        :type target: integer
        """
        self.near_graph.add_edge(source, target, weight = weight, type='near')