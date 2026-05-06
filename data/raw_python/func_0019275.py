def _guessunit(self):
        """Guess the unit of the period as the largest one, which results in
        an integer duration.
        """
        if not self.days % 1:
            return 'd'
        elif not self.hours % 1:
            return 'h'
        elif not self.minutes % 1:
            return 'm'
        elif not self.seconds % 1:
            return 's'
        else:
            raise ValueError(
                'The stepsize is not a multiple of one '
                'second, which is not allowed.')