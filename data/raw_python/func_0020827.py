def target_power(self):
        """Setting this to `True` will activate the power pins (4 and 6). If
        set to `False` the power will be deactivated.

        Raises an :exc:`IOError` if the hardware adapter does not support
        the switchable power pins.
        """
        ret = api.py_aa_target_power(self.handle, TARGET_POWER_QUERY)
        _raise_error_if_negative(ret)
        return ret