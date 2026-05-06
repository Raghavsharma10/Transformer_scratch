def EXPIRING_TOKEN_LIFESPAN(self):
        """
        Return the allowed lifespan of a token as a TimeDelta object.

        Defaults to 30 days.
        """
        try:
            val = settings.EXPIRING_TOKEN_LIFESPAN
        except AttributeError:
            val = timedelta(days=30)

        return val