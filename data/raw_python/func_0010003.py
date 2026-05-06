def generate_private_key(self):
        """
        Generates a private key of key_length bits and attaches it to the object as the __private_key variable.

        :return: void
        :rtype: void
        """
        key_length = self.key_length // 8 + 8
        key = 0

        try:
            key = int.from_bytes(rng(key_length), byteorder='big')
        except:
            key = int(hex(rng(key_length)), base=16)

        self.__private_key = key