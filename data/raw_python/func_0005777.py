def _expand_var(self, in_string, available_variables):
        """Expand variable to its corresponding value in_string

        :param string variable: variable name
        :param value: value to replace with
        :param string in_string: the string to replace in
        """
        instances = self._get_instances(in_string)
        for instance in instances:
            for name, value in available_variables.items():
                variable_string = self._get_variable_string(name)
                if instance == variable_string:
                    in_string = in_string.replace(variable_string, value)
        return in_string