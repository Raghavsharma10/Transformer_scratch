def set_password(self, raw_password):
        """
        Kullanıcı şifresini encrypt ederek set eder.

        Args:
            raw_password (str)
        """
        self.password = pbkdf2_sha512.encrypt(raw_password, rounds=10000,
                                              salt_size=10)