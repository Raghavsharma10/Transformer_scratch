def primary_parameters_complete(self):
        """True/False flag that indicates wheter the values of all primary
        parameters are defined or not."""
        for primpar in self._PRIMARY_PARAMETERS.values():
            if primpar.__get__(self) is None:
                return False
        return True