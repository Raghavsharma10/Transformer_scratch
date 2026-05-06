def set_load_resistance(self, resistance):
        """
        Changes load to resistance mode and sets resistance value.
        Rounds to nearest 0.01 Ohms

        :param resistance: Load Resistance in Ohms (0-500 ohms)
        :return: None
        """
        new_val = int(round(resistance * 100))
        if not 0 <= new_val <= 50000:
            raise ValueError("Load Resistance should be between 0-500 ohms")
        self._load_mode = self.SET_TYPE_RESISTANCE
        self._load_value = new_val
        self.__set_parameters()