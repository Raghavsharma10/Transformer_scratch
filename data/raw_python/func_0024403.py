def shift(self, direction):
        """
        Shifts in direction provided by ``Direction`` enum.

        :type: direction: Direction
        :rtype: Location
        """
        try:
            if direction == Direction.UP:
                return self.shift_up()
            elif direction == Direction.DOWN:
                return self.shift_down()
            elif direction == Direction.RIGHT:
                return self.shift_right()
            elif direction == Direction.LEFT:
                return self.shift_left()
            else:
                raise IndexError("Invalid direction {}".format(direction))
        except IndexError as e:
            raise IndexError(e)