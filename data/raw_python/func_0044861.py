def power_level(self, val):
        """Sets the power level according to the index or string.
        0 = High
        1 = HighLow
        2 = LowHigh
        3 = Low"""
        if val in [0, 1, 2, 3]:
            self.pdx.PowerLevel = val
        elif type(val) is str:
            if val.lower() == "high":
                self.pdx.PowerLevel = 0
            elif val.lower() == "highlow":
                self.pdx.PowerLevel = 1
            elif val.lower() == "lowhigh":
                self.pdx.PowerLevel = 2
            elif val.lower() == "low":
                self.pdx.PowerLevel = 3
        else:
            raise ValueError("Not a valid power level")