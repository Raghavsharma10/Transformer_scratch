def square_in_front(self, location=None):
        """
        Finds square directly in front of Pawn

        :type: location: Location
        :rtype: Location
        """
        location = location or self.location
        return location.shift_up() if self.color == color.white else location.shift_down()