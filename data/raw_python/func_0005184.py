def _pct_diff(self, best, other):
        """ Calculates and colorizes the percent difference between @best
            and @other
        """
        return colorize("{}%".format(
            round(((best-other)/best)*100, 2)).rjust(10), "red")