def close(self):
        """Closes the gpib transport."""
        if self._device is not None:
            ibsta = self._lib.ibonl(self._device, 0)
            self._check_status(ibsta)
            self._device = None