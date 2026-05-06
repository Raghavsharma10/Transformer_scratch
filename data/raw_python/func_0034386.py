def validate(self):
        """
        Execute the code once to get it's results (to be used in function validation). Compare the result to the
        first function in the group.
        """
        validation_code = self.setup_src + '\nvalidation_result = ' + self.stmt
        validation_scope = {}
        exec(validation_code, validation_scope)
        # Store the result in the first function in the group.
        if len(self.groups[self.group]) == 1:
            self.result = validation_scope['validation_result']
            logging.info('PyPerform: Validating group "{b.group}" against function "{b.callable.__name__}"'
                         .format(b=self))
        else:
            compare_against_benchmark = self.groups[self.group][0]
            test = [benchmark.result_validation for benchmark in self.groups[self.group]]
            if not all(test):
                raise ValueError('All functions within a group must have the same validation flag.')
            compare_result = compare_against_benchmark.result
            if self.validation_func:
                results_are_valid = self.validation_func(compare_result, validation_scope['validation_result'])
            else:
                results_are_valid = compare_result == validation_scope['validation_result']
            if results_are_valid:
                logging.info('PyPerform: Validating {}......PASSED!'.format(self.callable.__name__))
            else:
                error = 'Results of functions {0} and {1} are not equivalent.\n{0}:\t {2}\n{1}:\t{3}'
                raise ValidationError(error.format(compare_against_benchmark.callable.__name__, self.callable.__name__,
                                          compare_result, validation_scope['validation_result']))