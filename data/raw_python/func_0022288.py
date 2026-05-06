def read(self, vals):
        """Read values.

        Args:
            vals (list): list of strings representing values

        """
        i = 0
        if len(vals[i]) == 0:
            self.leapyear_observed = None
        else:
            self.leapyear_observed = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.daylight_saving_start_day = None
        else:
            self.daylight_saving_start_day = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.daylight_saving_end_day = None
        else:
            self.daylight_saving_end_day = vals[i]
        i += 1
        count = int(vals[i])
        i += 1
        for _ in range(count):
            obj = Holiday()
            obj.read(vals[i:i + obj.field_count])
            self.add_holiday(obj)
            i += obj.field_count