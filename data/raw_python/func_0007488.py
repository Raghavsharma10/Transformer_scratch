def expires_in(self):
        """
        Returns the time until the token expires.

        :return: The remaining time until expiration in seconds or 0 if the
                 token has expired.
        """
        time_left = self.expires_at - int(time.time())

        if time_left > 0:
            return time_left
        return 0