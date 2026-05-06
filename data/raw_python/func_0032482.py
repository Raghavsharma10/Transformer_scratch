def scan_temperature(self, measure, temperature, rate, delay=1):
        """Performs a temperature scan.

        Measures until the target temperature is reached.

        :param measure: A callable called repeatedly until stability at target
            temperature is reached.
        :param temperature: The target temperature in kelvin.
        :param rate: The sweep rate in kelvin per minute.
        :param delay: The time delay between each call to measure in seconds.

        """
        if not hasattr(measure, '__call__'):
            raise TypeError('measure parameter not callable.')

        self.set_temperature(temperature, rate, 'no overshoot', wait_for_stability=False)
        start = datetime.datetime.now()
        while True:
            # The PPMS needs some time to update the status code, we therefore ignore it for 10s.
            if (self.system_status['temperature'] == 'normal stability at target temperature' and
                (datetime.datetime.now() - start > datetime.timedelta(seconds=10))):
                break
            measure()
            time.sleep(delay)