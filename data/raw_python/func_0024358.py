def forward_moves(self, position):
        """
        Finds possible moves one step and two steps in front
        of Pawn.

        :type: position: Board
        :rtype: list
        """
        if position.is_square_empty(self.square_in_front(self.location)):
            """
            If square in front is empty add the move
            """
            if self.would_move_be_promotion():
                for move in self.create_promotion_moves(notation_const.PROMOTE):
                    yield move
            else:
                yield self.create_move(end_loc=self.square_in_front(self.location),
                                       status=notation_const.MOVEMENT)

            if self.on_home_row() and \
                    position.is_square_empty(self.two_squares_in_front(self.location)):
                """
                If pawn is on home row and two squares in front of the pawn is empty
                add the move
                """
                yield self.create_move(
                    end_loc=self.square_in_front(self.square_in_front(self.location)),
                    status=notation_const.MOVEMENT
                )