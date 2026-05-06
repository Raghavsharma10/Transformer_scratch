def set_load_current(self, current_amps):
        """
        Changes load to current mode and sets current value.
        Rounds to nearest mA.

        :param current_amps: Current in Amps (0-30A)
        :return: None
        """
        new_val = int(round(current_amps * 1000))
        if not 0 <= new_val <= 30000:
            raise ValueError("Load Current should be between 0-30A")
        self._load_mode = self.SET_TYPE_CURRENT
        self._load_value = new_val
        self.__set_parameters()