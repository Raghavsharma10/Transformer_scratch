def loc_adjacent_to_opponent_king(self, location, position):
        """
        Finds if 2 kings are touching given the position of one of the kings.

        :type: location: Location
        :type: position: Board
        :rtype: bool
        """
        for fn in self.cardinal_directions:
            try:
                if isinstance(position.piece_at_square(fn(location)), King) and \
                        position.piece_at_square(fn(location)).color != self.color:
                    return True

            except IndexError:
                pass

        return False