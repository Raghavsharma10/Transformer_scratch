def addUnexpectedSuccess(self, test):
        """Register a test that passed unexpectedly.

        Parameters
        ----------
        test : unittest.TestCase
            The test that has completed.

        """
        result = self._handle_result(
            test, TestCompletionStatus.unexpected_success)
        self.unexpectedSuccesses.append(result)