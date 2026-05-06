def save(self, path, check=True):
        """Save WeatherData in EPW format to path.

        Args:
            path (str): path where EPW file should be saved

        """
        with open(path, 'w') as f:
            if check:
                if ("LOCATION" not in self._data or
                        self._data["LOCATION"] is None):
                    raise ValueError('location is not valid.')
                if ("DESIGN CONDITIONS" not in self._data or
                        self._data["DESIGN CONDITIONS"] is None):
                    raise ValueError('design_conditions is not valid.')
                if ("TYPICAL/EXTREME PERIODS" not in self._data or
                        self._data["TYPICAL/EXTREME PERIODS"] is None):
                    raise ValueError(
                        'typical_or_extreme_periods is not valid.')
                if ("GROUND TEMPERATURES" not in self._data or
                        self._data["GROUND TEMPERATURES"] is None):
                    raise ValueError('ground_temperatures is not valid.')
                if ("HOLIDAYS/DAYLIGHT SAVINGS" not in self._data or
                        self._data["HOLIDAYS/DAYLIGHT SAVINGS"] is None):
                    raise ValueError(
                        'holidays_or_daylight_savings is not valid.')
                if ("COMMENTS 1" not in self._data or
                        self._data["COMMENTS 1"] is None):
                    raise ValueError('comments_1 is not valid.')
                if ("COMMENTS 2" not in self._data or
                        self._data["COMMENTS 2"] is None):
                    raise ValueError('comments_2 is not valid.')
                if ("DATA PERIODS" not in self._data or
                        self._data["DATA PERIODS"] is None):
                    raise ValueError('data_periods is not valid.')
            if ("LOCATION" in self._data and
                    self._data["LOCATION"] is not None):
                f.write(self._data["LOCATION"].export() + "\n")
            if ("DESIGN CONDITIONS" in self._data and
                    self._data["DESIGN CONDITIONS"] is not None):
                f.write(self._data["DESIGN CONDITIONS"].export() + "\n")
            if ("TYPICAL/EXTREME PERIODS" in self._data and
                    self._data["TYPICAL/EXTREME PERIODS"] is not None):
                f.write(self._data["TYPICAL/EXTREME PERIODS"].export() + "\n")
            if ("GROUND TEMPERATURES" in self._data and
                    self._data["GROUND TEMPERATURES"] is not None):
                f.write(self._data["GROUND TEMPERATURES"].export() + "\n")
            if ("HOLIDAYS/DAYLIGHT SAVINGS" in self._data and
                    self._data["HOLIDAYS/DAYLIGHT SAVINGS"] is not None):
                f.write(
                    self._data["HOLIDAYS/DAYLIGHT SAVINGS"].export() +
                    "\n")
            if ("COMMENTS 1" in self._data and
                    self._data["COMMENTS 1"] is not None):
                f.write(self._data["COMMENTS 1"].export() + "\n")
            if ("COMMENTS 2" in self._data and
                    self._data["COMMENTS 2"] is not None):
                f.write(self._data["COMMENTS 2"].export() + "\n")
            if ("DATA PERIODS" in self._data and
                    self._data["DATA PERIODS"] is not None):
                f.write(self._data["DATA PERIODS"].export() + "\n")
            for item in self._data["WEATHER DATA"]:
                f.write(item.export(False) + "\n")