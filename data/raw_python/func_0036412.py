def update_status(self, retry_count=2):
        """
        Updates current values from load.
        Must be called to get latest values for the following properties of class:
          current
          voltage
          power
          max current
          max power
          resistance
          local_control
          load_on
          wrong_polarity
          excessive_temp
          excessive_voltage
          excessive_power

        :param retry_count: Number of times to ignore IOErrors and retry update
        :return: None
        """
        # I think retry should be in here.
        # Throw exceptions in __update_status and handle here
        cur_count = max(retry_count, 0)
        while cur_count >= 0:
            try:
                self.__update_status()
            except IOError as err:
                if self.print_errors:
                    print("IOError: {}".format(err))
            else:
                if not self.__is_valid_checksum(self.__in_buffer.raw):
                    if self.print_errors:
                        raise IOError("Checksum validation failed.")
                values = self.STRUCT_READ_VALUES_IN.unpack_from(self.__in_buffer, self.OFFSET_FRONT)
                (self._current,
                 self._voltage,
                 self._power,
                 self._max_current,
                 self._max_power,
                 self._resistance,
                 output_state) = values[3:-1]

                self._remote_control = (output_state & 0b00000001) > 0
                self._load_on = (output_state & 0b00000010) > 0
                self.wrong_polarity = (output_state & 0b00000100) > 0
                self.excessive_temp = (output_state & 0b00001000) > 0
                self.excessive_voltage = (output_state & 0b00010000) > 0
                self.excessive_power = (output_state & 0b00100000) > 0
                return None
            cur_count -= 1
        raise IOError("Retry count exceeded with serial IO.")