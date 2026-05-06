def is_bored_of(self, board):
        """Return whether the simulation is probably in a loop.

        This is a stochastic guess. Basically, it detects whether the
        simulation has had the same number of cells a lot lately. May have
        false positives (like if you just have a screen full of gliders) or
        take awhile to catch on sometimes. I've even seen it totally miss the
        boat once. But it's simple and fast.

        """
        self.iteration += 1
        if len(board) == self.num:
            self.times += 1
        is_bored = self.times > self.REPETITIONS
        if self.iteration > self.REPETITIONS * self.PATTERN_LENGTH or is_bored:
            # A little randomness in case things divide evenly into each other:
            self.iteration = randint(-2, 0)
            self.num = len(board)
            self.times = 0
        return is_bored