def build_variables(self, variable_placeholders):
        """
        :param variables: The list of vertices/edges to return
        :return: a dict where the keys are the names of the variables to return,
                 the values are the JSON of the properties of these variables
        """
        variables = self.__substitute_names_in_list(variable_placeholders)
        attributes = {}
        for i, variable in enumerate(variables):
            placeholder_name = variable_placeholders[i]
            try:
                vertices = self.g.vs.select(name=variable)
                attributes[placeholder_name] = vertices[0].attributes()
            except:
                pass
        for i, variable in enumerate(variables):
            placeholder_name = variable_placeholders[i]
            try:
                edges = self.g.es.select(name=variable)
                edge_attr = edges[0].attributes()
                attributes[placeholder_name] = edge_attr
            except:
                pass
        for i, variable in enumerate(variables):
            placeholder_name = variable_placeholders[i]
            try:
                attributes[placeholder_name] = self.match_info[placeholder_name]
            except:
                pass
        return attributes