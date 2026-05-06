def _string_to_byte_list(self, data):
        """
        Creates a hex digest of the input string given to create the image,
        if it's not already hexadecimal

        Returns:
            Length 16 list of rgb value range integers
            (each representing a byte of the hex digest)
        """
        bytes_length = 16

        m = self.digest()
        m.update(str.encode(data))
        hex_digest = m.hexdigest()

        return list(int(hex_digest[num * 2:num * 2 + 2], bytes_length)
                    for num in range(bytes_length))