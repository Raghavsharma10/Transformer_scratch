def scan_temperature_old(self, measure, temperature, rate, delay=1):
        """Performs a temperature scan.

        Measures until the target temperature is reached.

        :param measure: A callable called repeatedly until stability at target
            temperature is reached.
        :param temperature: The target temperature in kelvin.
        :param rate: The sweep rate in kelvin per minute.
        :param delay: The time delay between each call to measure in seconds.

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
        while self.activity == 'sweep':
            measure()
            time.sleep(delay)