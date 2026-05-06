def __set_checksum(self):
        """
        Sets the checksum on the last byte of buffer,
        based on values in the buffer
        :return: None
        """
        checksum = self.__get_checksum(self.__out_buffer.raw)
        self.STRUCT_CHECKSUM.pack_into(self.__out_buffer, self.OFFSET_CHECKSUM, checksum)