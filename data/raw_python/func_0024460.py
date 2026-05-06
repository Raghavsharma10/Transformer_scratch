def all_possible_moves(self, input_color):
        """
        Checks if all the possible moves has already been calculated
        and is stored in `possible_moves` dictionary. If not, it is calculated
        with `_calc_all_possible_moves`.
        
        :type: input_color: Color
        :rtype: list
        """
        position_tuple = self.position_tuple
        if position_tuple not in self.possible_moves:
            self.possible_moves[position_tuple] = tuple(self._calc_all_possible_moves(input_color))

        return self.possible_moves[position_tuple]