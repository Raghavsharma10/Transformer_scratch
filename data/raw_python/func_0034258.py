def set(self, code):
        """
        Executes the code and apply it to the self.g

        :param code: the LISP code to execute
        :return: True/False, depending on the result of the LISP code
        """
        if self.update:
            self.vertices_substitution_dict, self.edges_substitution_dict, self.match_info\
                = self.match.get_variables_substitution_dictionaries(self.g, self.matching_graph)
        try:
            self.matching_graph = self.__apply_code_to_graph(code, self.matching_graph)
        except:
            pass
        try:
            code = self.__substitute_names_in_code(code)
            self.g = self.__apply_code_to_graph(code, self.g)
        except:
            pass
        return True