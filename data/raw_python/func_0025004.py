def _create_secret(self, length=12):
        """
        Use a cryptograhically-secure Pseudorandom number generator for picking
        a combination of letters, digits, and punctuation to be our secret.

        :param length: how long to make the secret (12 seems ok most of the time)

        """
        # Charset will have 64 +- characters
        charset = string.digits + string.ascii_letters + '+-'
        return "".join(random.SystemRandom().choice(charset) for _ in
                range(length))