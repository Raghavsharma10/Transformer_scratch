def _api_rate_limit_exceeded(self, api_call, window=60):
        """
        We want to keep track of the last time we sent a request to the NewRelic
        API, but only for certain operations. This method will dynamically add
        an attribute to the Client class with a unix timestamp with the name of
        the API api_call we make so that we can check it later.  We return the
        amount of time until we can perform another API call so that appropriate
        waiting can be implemented.
        """
        current = datetime.datetime.now()
        try:
            previous = getattr(self, api_call.__name__ + "_window")
            # Force the calling of our property so we can
            # handle not having set it yet.
            previous.__str__
        except AttributeError:
            now = datetime.datetime.now()
            outside_window = datetime.timedelta(seconds=window+1)
            previous = now - outside_window

        if current - previous > datetime.timedelta(seconds=window):
            setattr(self, api_call.__name__ + "_window", current)
        else:
            timeout = window - (current - previous).seconds
            raise NewRelicApiRateLimitException(str(timeout))