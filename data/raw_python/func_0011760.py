def heater_level(self, value):
        """Verifies that the heater_level is between 0 and heater_segments.
           Can only be called when freshroastsr700 object is initialized
           with ext_sw_heater_drive=True. Will throw RoasterValueError
           otherwise."""
        if self._ext_sw_heater_drive:
            if value not in range(0, self._heater_bangbang_segments+1):
                raise exceptions.RoasterValueError
            self._heater_level.value = value
        else:
            raise exceptions.RoasterValueError