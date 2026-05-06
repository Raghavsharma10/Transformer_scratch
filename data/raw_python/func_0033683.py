def set_light_state(self, hue, saturation, brightness, kelvin,
                        bulb=ALL_BULBS, timeout=None):
        """
        Sets the light state of one or more bulbs.

        Hue is a float from 0 to 360, saturation and brightness are floats from
        0 to 1, and kelvin is an integer.
        """
        raw_hue = int((hue % 360) / 360.0 * 0xffff) & 0xffff
        raw_sat = int(saturation * 0xffff) & 0xffff
        raw_bright = int(brightness * 0xffff) & 0xffff
        return self.set_light_state_raw(raw_hue, raw_sat, raw_bright, kelvin,
                                        bulb, timeout)