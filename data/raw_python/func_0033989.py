def query(self, string, repeat_n_times=None):
        """
        This method performs the operations onto self.g

        :param string: The list of operations to perform. The sequences of commands should be separated by a semicolon
                       An example might be
                         CREATE {'tag': 'PERSON', 'text': 'joseph'}(v1), {'relation': 'LIVES_AT'}(v1,v2),
                                {'tag': 'PLACE', 'text': 'London'}(v2)
                         MATCH {}(_a), {'relation': 'LIVES_AT'}(_a,_b), {}(_b)
                           WHERE (= (get _a "text") "joseph")
                         RETURN _a,_b;
        :param repeat_n_times: The maximum number of times the graph is queried. It sets the maximum length of
                               the return list. If None then the value is set by the function
                               self.__determine_how_many_times_to_repeat_query(string)

        :return: If the RETURN command is called with a list of variables names, a list of JSON with
                 the corresponding properties is returned. If the RETURN command is used alone, a list with the entire
                 graph is returned. Otherwise it returns an empty list
        """
        if not repeat_n_times:
            repeat_n_times = self.__determine_how_many_times_to_repeat_query(string)
        lines = self.__get_command_lines(string)
        return_list = []
        for line in lines:
            lst = self.__query_n_times(line, repeat_n_times)
            if lst and lst[0]:
                return_list = lst
        return return_list