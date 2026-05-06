def add(self, src):
        """ store an audio file to storage dir

        :param src: audio file path
        :return: checksum value
        """
        if not audio.get_type(src):
            raise TypeError('The type of this file is not supported.')

        return super().add(src)