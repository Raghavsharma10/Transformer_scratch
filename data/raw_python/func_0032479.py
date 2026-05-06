def move(self, position, slowdown=0):
        """Move to the specified sample position.

        :param position: The target position.
        :param slowdown: The slowdown code, an integer in the range 0 to 14,
            used to scale the stepper motor speed. 0, the default, is the
            fastest rate and 14 the slowest.

        """
        cmd = 'MOVE', [Float, Integer, Integer(min=0, max=14)]
        self._write(cmd, position, 0, slowdown)