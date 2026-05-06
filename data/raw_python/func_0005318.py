def check_bearer_validity(self, token: dict, connect_mtd) -> dict:
        """Check API Bearer token validity.

        Isogeo ID delivers authentication bearers which are valid during
        a certain time. So this method checks the validity of the token
        with a 30 mn anticipation limit, and renews it if necessary.
        See: http://tools.ietf.org/html/rfc6750#section-2

        FI: 24h = 86400 seconds, 30 mn = 1800, 5 mn = 300

        :param tuple token: auth bearer to check.
         Structure: (bearer, expiration_date)
        :param isogeo_pysdk.connect connect_mtd: method herited
         from Isogeo PySDK to get new bearer
        """
        warnings.warn(
            "Method is now executed as a decorator within the main SDK class. Will be removed in future versions.",
            DeprecationWarning,
        )
        if datetime.now() < token.get("expires_at"):
            token = connect_mtd
            logging.debug("Token was about to expire, so has been renewed.")
        else:
            logging.debug("Token is still valid.")
            pass

        # end of method
        return token