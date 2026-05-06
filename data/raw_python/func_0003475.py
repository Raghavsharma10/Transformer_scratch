def original_failure(self):
        """
        Return the underlying Failure object, if the result is an error.

        If no result is yet available, or the result was not an error, None is
        returned.

        This method is useful if you want to get the original traceback for an
        error result.
        """
        try:
            result = self._result(0.0)
        except TimeoutError:
            return None
        if isinstance(result, Failure):
            return result
        else:
            return None