def get_details(self):
        """ Overrides the method in Failure so as to add a few details about the wrapped function and outcome """

        if isinstance(self.validation_outcome, Exception):
            if isinstance(self.validation_outcome, Failure):
                # do not say again what was the value, it is already mentioned inside :)
                end_str = ''
            else:
                end_str = ' for value [{value}]'.format(value=self.wrong_value)

            contents = 'Function [{wrapped}] raised [{exception}: {details}]{end}.' \
                       ''.format(wrapped=get_callable_name(self.wrapped_func),
                                 exception=type(self.validation_outcome).__name__, details=self.validation_outcome,
                                 end=end_str)
        else:
            contents = 'Function [{wrapped}] returned [{result}] for value [{value}].' \
                       ''.format(wrapped=get_callable_name(self.wrapped_func), result=self.validation_outcome,
                                 value=self.wrong_value)

        return contents