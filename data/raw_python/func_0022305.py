def read(self, vals):
        """Read values.

        Args:
            vals (list): list of strings representing values

        """
        i = 0
        count = int(vals[i])
        i += 1
        for _ in range(count):
            obj = DataPeriod()
            obj.read(vals[i:i + obj.field_count])
            self.add_data_period(obj)
            i += obj.field_count