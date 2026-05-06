def addSkip(self, test, reason):
        """Register that a test that was skipped.

        Parameters
        ----------
        test : unittest.TestCase
            The test that has completed.
        reason : str
            The reason the test was skipped.

        """
        result = self._handle_result(
            test, TestCompletionStatus.skipped, message=reason)
        self.skipped.append(result)