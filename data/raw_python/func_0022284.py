def read(self, vals):
        """Read values.

        Args:
            vals (list): list of strings representing values

        """
        i = 0
        if len(vals[i]) == 0:
            self.holiday_name = None
        else:
            self.holiday_name = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.holiday_day = None
        else:
            self.holiday_day = vals[i]
        i += 1