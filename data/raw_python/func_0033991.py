def __get_action_graph_pairs_from_query(self, query):
        """
        Splits the query into command/argument pairs, for example [("MATCH","{}(_a))", ("RETURN","_a")]

        :param query: The string with the list of commands
        :return: the command/argument pairs
        """
        import re

        query = convert_special_characters_to_spaces(query)
        graph_list = re.split('|'.join(self.action_list), query)
        query_list_positions = [query.find(graph) for graph in graph_list]
        query_list_positions = query_list_positions
        query_list_positions = query_list_positions
        action_list = [query[query_list_positions[i] + len(graph_list[i]):query_list_positions[i + 1]].strip()
                       for i in range(len(graph_list) - 1)]
        graph_list = graph_list[1:]
        return zip(action_list, graph_list)