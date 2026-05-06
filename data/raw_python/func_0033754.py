def get_user_data_triplet(self, base64=False):
        """( <_user>, <_password verifier>, <salt> )

        :param base64:
        :rtype: tuple
        """
        salt = self.generate_salt()
        verifier = self.get_common_password_verifier(self.get_common_password_hash(salt))

        verifier = value_encode(verifier, base64)
        salt = value_encode(salt, base64)

        return self._user, verifier, salt