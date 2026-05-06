def scan_field(self, measure, target, rate, delay=1):
        """Performs a field scan.

        Measures until the target field is reached.

        :param measure: A callable called repeatedly until stability at the
            target field is reached.
        :param field: The target field in Tesla.
        :param rate: The field rate in tesla per minute.
        :param delay: The time delay between each call to measure in seconds.

        :raises TypeError: if measure parameter is not callable.

        """
        if not hasattr(measure, '__call__'):
            raise TypeError('measure parameter not callable.')
        self.activity = 'hold'
        self.field.target = target
        self.field.sweep_rate = rate
        self.activity = 'to setpoint'
        while self.status['mode'] != 'at rest':
            measure()
            time.sleep(delay)