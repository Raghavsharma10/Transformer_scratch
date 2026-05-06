def convert(self, value):
        """ Convert the specified value to the type of the option.

        Args:
            value: The value that should be converted.

        Returns:
            The value with the type given by the option.
        """
        if self._type is str:
            return str(value)
        elif self._type is int:
            try:
                return int(value)
            except (UnicodeError, ValueError):
                raise WorkflowArgumentError('Cannot convert {} to int'.format(value))
        elif self._type is float:
            try:
                return float(value)
            except (UnicodeError, ValueError):
                raise WorkflowArgumentError('Cannot convert {} to float'.format(value))
        elif self._type is bool:
            if isinstance(value, bool):
                return bool(value)
            value = value.lower()
            if value in ('true', '1', 'yes', 'y'):
                return True
            elif value in ('false', '0', 'no', 'n'):
                return False
            raise WorkflowArgumentError('Cannot convert {} to bool'.format(value))
        else:
            return value