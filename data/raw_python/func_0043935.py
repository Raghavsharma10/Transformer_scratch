def ensure_hexadecimal_string(self, value, command=None):
        """
        Make sure the given value is a hexadecimal string.

        :param value: The value to check (a string).
        :param command: The command that produced the value (a string or :data:`None`).
        :returns: The validated hexadecimal string.
        :raises: :exc:`~exceptions.ValueError` when `value` is not a hexadecimal string.
        """
        if not HEX_PATTERN.match(value):
            msg = "Expected a hexadecimal string, got '%s' instead!"
            if command:
                msg += " ('%s' gave unexpected output)"
                msg %= (value, command)
            else:
                msg %= value
            raise ValueError(msg)
        return value