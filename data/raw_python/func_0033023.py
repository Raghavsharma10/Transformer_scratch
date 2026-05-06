def set_temperature(self, temperature, rate, delay=1):
        """Performs a temperature scan.

        Measures until the target temperature is reached.

        :param measure: A callable called repeatedly until stability at target
            temperature is reached.
        :param temperature: The target temperature in kelvin.
        :param rate: The sweep rate in kelvin per minute.
        :param delay: The time delay between each call to measure in seconds.

        """
        self.scan_temperature(lambda: None, temperature, rate, delay)