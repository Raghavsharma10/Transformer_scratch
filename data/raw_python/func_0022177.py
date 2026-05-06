def read(self, vals):
        """Read values.

        Args:
            vals (list): list of strings representing values

        """
        i = 0
        if len(vals[i]) == 0:
            self.city = None
        else:
            self.city = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.state_province_region = None
        else:
            self.state_province_region = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.country = None
        else:
            self.country = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.source = None
        else:
            self.source = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.wmo = None
        else:
            self.wmo = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.latitude = None
        else:
            self.latitude = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.longitude = None
        else:
            self.longitude = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.timezone = None
        else:
            self.timezone = vals[i]
        i += 1
        if len(vals[i]) == 0:
            self.elevation = None
        else:
            self.elevation = vals[i]
        i += 1