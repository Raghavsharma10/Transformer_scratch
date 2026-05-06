def add_graph(self, rhs_graph):
        """
        Adds a graph to self.g

        :param rhs_graph: the graph to add
        :return: itself
        """
        rhs_graph = self.__substitute_names_in_graph(rhs_graph)
        self.g = self.__merge_graphs(self.g, rhs_graph)
        return self