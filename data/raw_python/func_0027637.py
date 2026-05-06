def get_what_txt(self):
        """
        Overrides the base behaviour defined in ValidationError in order to add details about the function.
        :return:
        """
        return 'input [{var}] for function [{func}]'.format(var=self.get_variable_str(),
                                                            func=self.validator.get_validated_func_display_name())