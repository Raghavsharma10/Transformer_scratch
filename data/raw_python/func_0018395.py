def _class(self):
        """Return the class I should be, spanning a continuum of goodness."""
        try:
            self._project_name()
        except ValueError:
            return MalformedReq
        if self._is_satisfied():
            return SatisfiedReq
        if not self._expected_hashes():
            return MissingReq
        if self._actual_hash() not in self._expected_hashes():
            return MismatchedReq
        return InstallableReq