def read(self, path):
        """Read EPW weather data from path.

        Args:
            path (str): path to read weather data from

        """
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                match_obj_name = re.search(r"^([A-Z][A-Z/ \d]+),", line)
                if match_obj_name is not None:
                    internal_name = match_obj_name.group(1)
                    if internal_name in self._data:
                        self._data[internal_name] = self._create_datadict(
                            internal_name)
                        data_line = line[len(internal_name) + 1:]
                        vals = data_line.strip().split(',')
                        self._data[internal_name].read(vals)
                else:
                    wd = WeatherData()
                    wd.read(line.strip().split(','))
                    self.add_weatherdata(wd)