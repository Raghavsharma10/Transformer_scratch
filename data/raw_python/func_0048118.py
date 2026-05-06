def weekly_growth(self, weeks):
        """Calculate the weekly growth in percentage, and rounds
        to one digit.
    
        Parameters
        ----------
        weeks
            Number of weeks to calculate growth over.

        Returns
        -------
        growth_factor
            A real number such that start * growth_factor** weeks = end.
    
    
        Examples
        -------
        >>> bench = DynamicExercise('Bench press', 100, 120, 3, 8)
        >>> bench.weekly_growth(8)
        2.3
        >>> bench.weekly_growth(4)
        4.7
        """
        start, end = self.start_weight, self.final_weight
        growth_factor = ((end / start) ** (1 / weeks) - 1) * 100
        return round(growth_factor, 1)