def __clear_in_buffer(self):
        """
        Zeros out the in buffer
        :return: None
        """
        self.__in_buffer.value = bytes(b'\0' * len(self.__in_buffer))