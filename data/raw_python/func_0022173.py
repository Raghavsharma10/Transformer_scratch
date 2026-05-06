def _bit_is_one(self, n, hash_bytes):
        """
        Check if the n (index) of hash_bytes is 1 or 0.
        """

        scale = 16  # hexadecimal

        if not hash_bytes[int(n / (scale / 2))] >> int(
                (scale / 2) - ((n % (scale / 2)) + 1)) & 1 == 1:
            return False
        return True