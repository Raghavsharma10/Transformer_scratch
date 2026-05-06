def execute(self, vertices_substitution_dict={}):
        """
        Executes the code

        :param vertices_substitution_dict: aliases of the variables in the code
        :return: True/False, depending on the result of the code (default is True)
        """

        if not self.code_strings:
            return True

        if vertices_substitution_dict:
            namespace = self.__substitute_names_in_namespace(self.namespace, vertices_substitution_dict)
        else:
            namespace = self.namespace
        try:
            self.__execute_code(self._compiled_ast_and_expr, namespace)
        except:
            pass
        return namespace['result']