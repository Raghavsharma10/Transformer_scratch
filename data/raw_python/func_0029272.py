def addExpectedFailure(self, test, exception):
        """Register that a test that failed and was expected to fail.

        Parameters
        ----------
        test : unittest.TestCase
            The test that has completed.
        exception : tuple
            ``exc_info`` tuple ``(type, value, traceback)``.

        """
        result = self._handle_result(
            test, TestCompletionStatus.expected_failure, exception=exception)
        self.expectedFailures.append(result)