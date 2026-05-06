def read(self, vals):
        """Read values.

        Args:
            vals (list): list of strings representing values

        """
        i = 0
        if len(vals[i]) == 0:
            self.typical_or_extreme_period_name = None
        else:
            self.typical_or_extreme_period_name = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.typical_or_extreme_period_type = None
        else:
            self.typical_or_extreme_period_type = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.period_start_day = None
        else:
            self.period_start_day = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.period_end_day = None
        else:
            self.period_end_day = vals[i]
        i += 1