def _callback(self, *dummy):
        """
        This gets called on any attempt to change the value
        """
        # retrieve the value from the Entry
        value = self._variable.get()
        # run the validation. Returns None if no good
        newvalue = self.validate(value)
        if newvalue is None:
            # Invalid: restores previously stored value
            # no checker run.
            self._variable.set(self.as_string())
        else:
            # Store new value
            self._value = coord.Angle(value, unit=self.unit)
            if self.checker:
                self.checker(*dummy)