def set_load_power(self, power_watts):
        """
        Changes load to power mode and sets power value.
        Rounds to nearest 0.1W.

        :param power_watts: Power in Watts (0-200)
        :return:
        """
        new_val = int(round(power_watts * 10))
        if not 0 <= new_val <= 2000:
            raise ValueError("Load Power should be between 0-200 W")
        self._load_mode = self.SET_TYPE_POWER
        self._load_value = new_val
        self.__set_parameters()