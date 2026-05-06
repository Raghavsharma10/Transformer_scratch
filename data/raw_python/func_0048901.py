def set_stdout(self, stream=None):
        """
        Set the logger standard output stream.

        :Parameters:
           #. stdout (None, stream): The standard output stream. If None, system standard
              output will be set automatically. Otherwise any stream with read and write
              methods can be passed
        """
        if stream is None:
            self.__stdout = sys.stdout
        else:
            assert hasattr(stream, 'read') and hasattr(stream, 'write'), "stdout stream is not valid"
            self.__stdout = stream
        # set stdout colors
        self.__stdoutFontFormat = self.__get_stream_fonts_attributes(stream)