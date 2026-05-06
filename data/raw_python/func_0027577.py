def get_variable_str(self):
        """
        Utility method to get the variable value or 'var_name=value' if name is not None.
        Note that values with large string representations will not get printed

        :return:
        """
        if self.var_name is None:
            prefix = ''
        else:
            prefix = self.var_name

        suffix = str(self.var_value)
        if len(suffix) == 0:
            suffix = "''"
        elif len(suffix) > self.__max_str_length_displayed__:
            suffix = ''

        if len(prefix) > 0 and len(suffix) > 0:
            return prefix + '=' + suffix
        else:
            return prefix + suffix