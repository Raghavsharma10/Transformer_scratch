def generate_public_key(self):
        """
        Generates public key.

        :return: void
        :rtype: void
        """
        self.public_key = pow(self.generator,
                              self.__private_key,
                              self.prime)