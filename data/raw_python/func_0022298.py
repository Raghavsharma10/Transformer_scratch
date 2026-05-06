def read(self, vals):
        """Read values.

        Args:
            vals (list): list of strings representing values

        """
        i = 0
        if len(vals[i]) == 0:
            self.number_of_records_per_hour = None
        else:
            self.number_of_records_per_hour = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.data_period_name_or_description = None
        else:
            self.data_period_name_or_description = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.data_period_start_day_of_week = None
        else:
            self.data_period_start_day_of_week = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.data_period_start_day = None
        else:
            self.data_period_start_day = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.data_period_end_day = None
        else:
            self.data_period_end_day = vals[i]
        i += 1