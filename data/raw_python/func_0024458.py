def material_advantage(self, input_color, val_scheme):
        """
        Finds the advantage a particular side possesses given a value scheme.

        :type: input_color: Color
        :type: val_scheme: PieceValues
        :rtype: double
        """

        if self.get_king(input_color).in_check(self) and self.no_moves(input_color):
            return -100

        if self.get_king(-input_color).in_check(self) and self.no_moves(-input_color):
            return 100

        return sum([val_scheme.val(piece, input_color) for piece in self])