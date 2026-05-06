def __send_buffer(self):
        """
        Sends the contents of self.__out_buffer to serial device
        :return: Number of bytes written
        """
        bytes_written = self.serial.write(self.__out_buffer.raw)
        if self.DEBUG_MODE:
            print("Wrote: '{}'".format(binascii.hexlify(self.__out_buffer.raw)))
        if bytes_written != len(self.__out_buffer):
            raise IOError("{} bytes written for output buffer of size {}".format(bytes_written,
                                                                                 len(self.__out_buffer)))
        return bytes_written