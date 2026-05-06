def trigger(self):
        """Triggers the device.

        The trigger method sens a GET(group execute trigger) command byte to
        the device.
        """
        ibsta = self._lib.ibtrg(self._device)
        self._check_status(ibsta)