def get_image(self, string, width, height, pad=0):
        """
          Byte representation of a PNG image
        """
        hex_digest_byte_list = self._string_to_byte_list(string)
        matrix = self._create_matrix(hex_digest_byte_list)
        return self._create_image(matrix, width, height, pad)