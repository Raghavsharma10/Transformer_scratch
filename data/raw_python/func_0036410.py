def __send_receive_buffer(self):
        """
        Performs a send of self.__out_buffer and then an immediate read into self.__in_buffer

        :return: None
        """
        self.__clear_in_buffer()
        self.__send_buffer()
        read_string = self.serial.read(len(self.__in_buffer))
        if self.DEBUG_MODE:
            print("Read: '{}'".format(binascii.hexlify(read_string)))
        if len(read_string) != len(self.__in_buffer):
            raise IOError("{} bytes received for input buffer of size {}".format(len(read_string),
                                                                                 len(self.__in_buffer)))
        if not self.__is_valid_checksum(read_string):
            raise IOError("Checksum validation failed on received data")
        self.__in_buffer.value = read_string