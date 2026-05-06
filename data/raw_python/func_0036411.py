def __set_parameters(self):
        """
        Sets Load Parameters from class values, including:
        Max Current, Max Power, Address, Load Mode, Load Value

        :return: None
        """
        self.__set_buffer_start(self.CMD_SET_PARAMETERS)
        # Can I send 0xFF as address to not change it each time?
        # Worry about writing to EEPROM or Flash with each address change.
        # Would then implement a separate address only change function.
        self.STRUCT_SET_PARAMETERS.pack_into(self.__out_buffer, self.OFFSET_PAYLOAD,
                                             self._max_current, self._max_power, self.address,
                                             self._load_mode, self._load_value)
        self.__set_checksum()
        self.__send_buffer()
        self.update_status()