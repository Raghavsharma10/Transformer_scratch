def eval(self, x):
        """This method returns the evaluation of the function with input x

        :param x: this is the input as a Long
        """
        aes = AES.new(self.key, AES.MODE_CFB, "\0" * AES.block_size)
        while True:
            nonce = 0
            data = KeyedPRF.pad(SHA256.new(str(x + nonce).encode()).digest(),
                                (number.size(self.range) + 7) // 8)
            num = self.mask & number.bytes_to_long(aes.encrypt(data))
            if (num < self.range):
                return num
            nonce += 1