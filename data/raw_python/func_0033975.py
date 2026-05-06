def get_variables_substitution_dictionaries(self, lhs_graph, rhs_graph):
        """
        Looks for sub-isomorphisms of rhs into lhs

        :param lhs_graph: The graph to look sub-isomorphisms into (the bigger graph)
        :param rhs_graph: The smaller graph
        :return: The list of matching names
        """
        if not rhs_graph:
            return {}, {}, {}
        self.matching_code_container.add_graph_to_namespace(lhs_graph)
        self.matching_code_container.add_graph_to_namespace(rhs_graph)
        return self.__collect_variables_that_match_graph(lhs_graph, rhs_graph)