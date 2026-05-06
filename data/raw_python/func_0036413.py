def set_program_sequence(self, array_program):
        """
        Sets program up in load.
        :param array_program: Populated Array3710Program object
        :return: None
        """
        self.__set_buffer_start(self.CMD_DEFINE_PROG_1_5)
        array_program.load_buffer_one_to_five(self.__out_buffer)
        self.__set_checksum()
        self.__send_buffer()

        self.__set_buffer_start(self.CMD_DEFINE_PROG_6_10)
        array_program.load_buffer_six_to_ten(self.__out_buffer)
        self.__set_checksum()
        self.__send_buffer()