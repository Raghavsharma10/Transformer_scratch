def set_light_state_raw(self, hue, saturation, brightness, kelvin,
                            bulb=ALL_BULBS, timeout=None):
        """
        Sets the (low-level) light state of one or more bulbs.
        """
        with _blocking(self.lock, self.light_state, self.light_state_event,
                       timeout):
            self.send(REQ_SET_LIGHT_STATE, bulb, 'xHHHHI',
                      hue, saturation, brightness, kelvin, 0)
            self.send(REQ_GET_LIGHT_STATE, ALL_BULBS, '')
        return self.light_state