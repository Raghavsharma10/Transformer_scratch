def write(self, reg, value):
        """Write raw byte value to the specified register

        :param reg: the register number (0-69, 250-255)
        :param value: byte value
        """
        # TODO: check reg: 0-69, 250-255
        self.__check_range('register_value', value)
        logger.debug("Write '%s' to register '%s'" %  (value, reg))
        self.__bus.write_byte_data(self.__address, reg, value)