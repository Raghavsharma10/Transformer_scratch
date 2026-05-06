def generate_shared_secret(self, other_public_key, echo_return_key=False):
        """
        Generates shared secret from the other party's public key.

        :param other_public_key: Other party's public key
        :type other_public_key: int
        :param echo_return_key: Echo return shared key
        :type bool
        :return: void
        :rtype: void
        """
        if self.verify_public_key(other_public_key) is False:
            raise MalformedPublicKey

        self.shared_secret = pow(other_public_key,
                                 self.__private_key,
                                 self.prime)

        shared_secret_as_bytes = self.shared_secret.to_bytes(self.shared_secret.bit_length() // 8 + 1, byteorder='big')

        _h = sha256()
        _h.update(bytes(shared_secret_as_bytes))

        self.shared_key = _h.hexdigest()

        if echo_return_key is True:
            return self.shared_key