def redefine_position(self, position):
        """Redefines the current position to the new position.

        :param position: The new position.

        """
        cmd = 'MOVE', [Float, Integer]
        self._write(cmd, position, 2)