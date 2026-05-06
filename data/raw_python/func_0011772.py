def _generate_packet(self):
        """Generates a packet based upon the current class variables. Note that
        current temperature is not sent, as the original application sent zeros
        to the roaster for the current temperature."""
        roaster_time = utils.seconds_to_float(self._time_remaining.value)
        packet = (
            self._header.value +
            self._temp_unit.value +
            self._flags.value +
            self._current_state.value +
            struct.pack(">B", self._fan_speed.value) +
            struct.pack(">B", int(round(roaster_time * 10.0))) +
            struct.pack(">B", self._heat_setting.value) +
            b'\x00\x00' +
            self._footer)

        return packet