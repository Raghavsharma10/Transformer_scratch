def reset(self, required=False):
        """
        Perform a reset and check for presence pulse.

        :param bool required: require presence pulse
        """
        reset = self._ow.reset()
        if required and reset:
            raise OneWireError("No presence pulse found. Check devices and wiring.")
        return not reset