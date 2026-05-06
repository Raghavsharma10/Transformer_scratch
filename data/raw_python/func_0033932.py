def add_line(self, string):
        """
        Adds a line to the LISP code to execute

        :param string: The line to add
        :return: None
        """
        self.code_strings.append(string)
        code = ''
        if len(self.code_strings) == 1:
            code = '(setv result ' + self.code_strings[0] + ')'
        if len(self.code_strings) > 1:
            code = '(setv result (and ' + ' '.join(self.code_strings) + '))'
        self._compiled_ast_and_expr = self.__compile_code(code_string=code)