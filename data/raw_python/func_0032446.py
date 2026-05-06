def _factory_default(self, confirm=False):
        """Resets the device to factory defaults.

        :param confirm: This function should not normally be used, to prevent
            accidental resets, a confirm value of `True` must be used.

        """
        if confirm is True:
            self._write(('DFLT', Integer), 99)
        else:
            raise ValueError('Reset to factory defaults was not confirmed.')