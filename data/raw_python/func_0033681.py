def set_power_state(self, is_on, bulb=ALL_BULBS, timeout=None):
        """
        Sets the power state of one or more bulbs.
        """
        with _blocking(self.lock, self.power_state, self.light_state_event,
                       timeout):
            self.send(REQ_SET_POWER_STATE,
                      bulb, '2s', '\x00\x01' if is_on else '\x00\x00')
            self.send(REQ_GET_LIGHT_STATE, ALL_BULBS, '')
        return self.power_state