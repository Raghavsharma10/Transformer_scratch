def __query_with_builder(self, string, builder):
        """
        Uses the builder in the argument to modify the graph, according to the commands in the string

        :param string: The single query to the database
        :return: The result of the RETURN operation
        """
        action_graph_pairs = self.__get_action_graph_pairs_from_query(string)
        for action, graph_str in action_graph_pairs:
            if action == 'RETURN' or action == '':
                return self.__return(graph_str, builder)
            try:
                self.action_dict[action](graph_str, builder)
            except MatchException:
                break
        return {}