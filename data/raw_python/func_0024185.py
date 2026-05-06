def play(self):
        """
        Starts game and returns one of 3 results . 
        Iterates between methods ``white_move()`` and
        ``black_move()`` until game ends. Each
        method calls the respective player's ``generate_move()``
        method.

        :rtype: int
        """
        colors = [lambda: self.white_move(), lambda: self.black_move()]
        colors = itertools.cycle(colors)

        while True:
            color_fn = next(colors)
            if game_state.no_moves(self.position):
                if self.position.get_king(color.white).in_check(self.position):
                    return 1

                elif self.position.get_king(color.black).in_check(self.position):
                    return 0

                else:
                    return 0.5

            color_fn()