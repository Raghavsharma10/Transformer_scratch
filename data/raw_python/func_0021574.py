def addError(self, test, error):
        """
        when a test raises an uncaught exception
        :param test:
        :param error:
        :return:
        """
        # test_dict will be None if startTest wasn't called (i.e. exception in setUp/setUpClass)
        # status=BROKEN
        if self.current_sample is not None:
            assertion_name = error[0].__name__
            error_msg = str(error[1]).split('\n')[0]
            error_trace = self._get_trace(error)
            self.current_sample.add_assertion(assertion_name)
            self.current_sample.set_assertion_failed(assertion_name, error_msg, error_trace)