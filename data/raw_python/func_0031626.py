def get_random_password(self, length=32, chars=None):
        """Helper function that gets a random password.

        :param length: The length of the random password.
        :type  length: int
        :param  chars: A string with characters to choose from. Defaults to all ASCII letters and digits.
        :type   chars: str
        """
        if chars is None:
            chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for x in range(length))