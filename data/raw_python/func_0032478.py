def beep(self, duration, frequency):
        """Generates a beep.

        :param duration: The duration in seconds, in the range 0.1 to 5.
        :param frequency: The frequency in Hz, in the range 500 to 5000.

        """
        cmd = 'BEEP', [Float(min=0.1, max=5.0), Integer(min=500, max=5000)]
        self._write(cmd, duration, frequency)