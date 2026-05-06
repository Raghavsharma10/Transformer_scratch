def _check_bearer_validity(decorated_func):
        """Check API Bearer token validity.

        Isogeo ID delivers authentication bearers which are valid during
        a certain time. So this decorator checks the validity of the token
        comparing with actual datetime (UTC) and renews it if necessary.
        See: http://tools.ietf.org/html/rfc6750#section-2

        :param decorated_func token: original function to execute after check
        """

        @wraps(decorated_func)
        def wrapper(self, *args, **kwargs):
            # compare token expiration date and ask for a new one if it's expired
            if datetime.now() < self.token.get("expires_at"):
                self.connect()
                logging.debug("Token was about to expire, so has been renewed.")
            else:
                logging.debug("Token is still valid.")
                pass

            # let continue running the original function
            return decorated_func(self, *args, **kwargs)

        return wrapper