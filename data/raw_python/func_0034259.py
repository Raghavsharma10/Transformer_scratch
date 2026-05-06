def delete_list(self, variables):
        """
        Deletes a list of vertices/edges from self.g

        :param variables: the names of the variables to delete
        :return:
        """
        variables = set(self.__substitute_names_in_list(variables))
        self.update = False
        self.g.delete_vertices(self.g.vs.select(name_in=variables))
        self.g.delete_edges(self.g.es.select(name_in=variables))