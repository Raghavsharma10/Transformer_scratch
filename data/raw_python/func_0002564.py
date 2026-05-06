def addFailure(self, result):
        """Add a failure to the result."""
        result.addFailure(self, (Exception, Exception(), None))
        # Since TAP will not provide assertion data, clean up the assertion
        # section so it is not so spaced out.
        test, err = result.failures[-1]
        result.failures[-1] = (test, "")