def move_to_limit(self, position):
        """Move to limit switch and define it as position.

        :param position: The new position of the limit switch.

        """
        cmd = 'MOVE', [Float, Integer]
        self._write(cmd, position, 1)