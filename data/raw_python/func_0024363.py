def add_one_en_passant_move(self, direction, position):
        """
        Yields en_passant moves in given direction if it is legal.

        :type: direction: function
        :type: position: Board
        :rtype: gen
        """
        try:
            if self._is_en_passant_valid(direction(self.location), position):
                yield self.create_move(
                    end_loc=self.square_in_front(direction(self.location)),
                    status=notation_const.EN_PASSANT
                )
        except IndexError:
            pass