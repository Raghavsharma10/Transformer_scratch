def _handle_result(self, test, status, exception=None, message=None):
        """Create a :class:`~.TestResult` and add it to this
        :class:`~ResultCollector`.

        Parameters
        ----------
        test : unittest.TestCase
            The test that this result will represent.
        status : haas.result.TestCompletionStatus
            The status of the test.
        exception : tuple
            ``exc_info`` tuple ``(type, value, traceback)``.
        message : str
            Optional message associated with the result (e.g. skip
            reason).

        """
        if self.buffer:
            stderr = self._stderr_buffer.getvalue()
            stdout = self._stdout_buffer.getvalue()
        else:
            stderr = stdout = None

        started_time = self._test_timing.get(self._testcase_to_key(test))
        if started_time is None and isinstance(test, ErrorHolder):
            started_time = datetime.utcnow()
        elif started_time is None:
            raise RuntimeError(
                'Missing test start! Please report this error as a bug in '
                'haas.')

        completion_time = datetime.utcnow()
        duration = TestDuration(started_time, completion_time)
        result = TestResult.from_test_case(
            test,
            status,
            duration=duration,
            exception=exception,
            message=message,
            stdout=stdout,
            stderr=stderr,
        )
        self.add_result(result)
        return result