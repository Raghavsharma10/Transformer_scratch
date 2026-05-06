def set_temperature_old(self, temperature, rate, wait_for_stability=True, delay=1):
        """Sets the temperature.

        .. note::

            For complex sweep sequences, checkout :attr:`ITC503.sweep_table`.

        :param temperature: The target temperature in kelvin.
        :param rate: The sweep rate in kelvin per minute.
        :param wait_for_stability: If wait_for_stability is `True`, the function
            call blocks until the target temperature is reached and stable.
        :param delay: The delay specifies the frequency how often the status is
            checked.

        """
        self.activity = 'hold'
        # Clear old sweep table
        self.sweep_table.clear()

        # Use current temperature as target temperature
        # and calculate sweep time.
        current_temperature = self.control_temperature
        sweep_time = abs((temperature - current_temperature) / rate)

        self.sweep_table[0] = temperature, sweep_time, 0.
        self.sweep_table[-1] = temperature, 0., 0.

        self.activity = 'sweep'
        if wait_for_stability:
            while self.activity == 'sweep':
                time.sleep(delay)