def _set_channel_gain(self, num):
        """
        Finish data transmission from HX711 by setting
        next required gain and channel

        Only called from the _read function.
        :param num: how often so do the set (1...3)
        :type num: int
        :return True on success
        :rtype bool
        """
        if not 1 <= num <= 3:
            raise AttributeError(
                """"num" has to be in the range of 1 to 3"""
            )

        for _ in range(num):
            logging.debug("_set_channel_gain called")
            start_counter = time.perf_counter()  # start timer now.
            GPIO.output(self._pd_sck, True)  # set high
            GPIO.output(self._pd_sck, False)  # set low
            end_counter = time.perf_counter()  # stop timer
            time_elapsed = float(end_counter - start_counter)
            # check if HX711 did not turn off...
            # if pd_sck pin is HIGH for 60 µs and more the HX 711 enters power down mode.
            if time_elapsed >= 0.00006:
                logging.warning(
                    'setting gain and channel took more than 60µs. '
                    'Time elapsed: {:0.8f}'.format(time_elapsed)
                )
                # hx711 has turned off. First few readings are inaccurate.
                # Despite this reading was ok and data can be used.
                result = self.get_raw_data(times=6)  # set for the next reading.
                if result is False:
                    raise GenericHX711Exception("channel was not set properly")
        return True