def start_program(self, turn_on_load=True):
        """
        Starts running programmed test sequence
        :return: None
        """
        self.__set_buffer_start(self.CMD_START_PROG)
        self.__set_checksum()
        self.__send_buffer()
        # Turn on Load if not on
        if turn_on_load and not self.load_on:
            self.load_on = True