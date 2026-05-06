def __set_buffer_start(self, command):
        """
        This sets the first three bytes and clears the other 23 bytes.
        :param command: Command Code to set
        :return: None
        """
        self.STRUCT_FRONT.pack_into(self.__out_buffer, self.OFFSET_FRONT, 0xAA, self.address, command)