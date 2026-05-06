def get_details(self):
        """ The function called to get the details appended to the help message when self.append_details is True """
        strval = str(self.wrong_value)
        if len(strval) > self.__max_str_length_displayed__:
            return '(Actual value is too big to be printed in this message)'
        else:
            return 'Wrong value: [{}]'.format(self.wrong_value)