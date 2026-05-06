def _read(self, max_tries=40):
        """
        - read the bit stream from HX711 and convert to an int value.
        - validates the acquired data
        :param max_tries: how often to try to get data
        :type max_tries: int
        :return raw data
        :rtype: int
        """
        # start by setting the pd_sck to false
        GPIO.output(self._pd_sck, False)
        # init the counter
        ready_counter = 0

        # loop until HX711 is ready
        # halt when maximum number of tires is reached
        while self._ready() is False:
            time.sleep(0.01)  # sleep for 10 ms before next try
            ready_counter += 1  # increment counter
            # check loop count
            # and stop when defined maximum is reached
            if ready_counter >= max_tries:
                logging.debug('self._read() not ready after 40 trials\n')
                return False

        data_in = 0  # 2's complement data from hx 711
        # read first 24 bits of data
        for i in range(24):
            # start timer
            start_counter = time.perf_counter()
            # request next bit from HX711
            GPIO.output(self._pd_sck, True)
            GPIO.output(self._pd_sck, False)
            # stop timer
            end_counter = time.perf_counter()
            time_elapsed = float(end_counter - start_counter)

            # check if the hx 711 did not turn off:
            # if pd_sck pin is HIGH for 60 us and more than the HX 711 enters power down mode.
            if time_elapsed >= 0.00006:
                logging.debug('Reading data took longer than 60µs. Time elapsed: {:0.8f}'.format(time_elapsed))
                return False

            # Shift the bits as they come to data_in variable.
            # Left shift by one bit then bitwise OR with the new bit.
            data_in = (data_in << 1) | GPIO.input(self._dout)

        if self.channel == 'A' and self.channel_a_gain == 128:
            self._set_channel_gain(num=1)  # send one bit
        elif self.channel == 'A' and self.channel_a_gain == 64:
            self._set_channel_gain(num=3)  # send three bits
        else:
            self._set_channel_gain(num=2)  # send two bits

        logging.debug('Binary value as it has come: ' + str(bin(data_in)))

        # check if data is valid
        # 0x800000 is the lowest
        # 0x7fffff is the highest possible value from HX711
        if data_in == 0x7fffff or data_in == 0x800000:
            logging.debug('Invalid data detected: ' + str(data_in))
            return False

        # calculate int from 2's complement
        signed_data = 0
        if (data_in & 0x800000):  # 0b1000 0000 0000 0000 0000 0000 check if the sign bit is 1. Negative number.
            signed_data = -((data_in ^ 0xffffff) + 1)  # convert from 2's complement to int
        else:  # else do not do anything the value is positive number
            signed_data = data_in

        logging.debug('Converted 2\'s complement value: ' + str(signed_data))

        return signed_data