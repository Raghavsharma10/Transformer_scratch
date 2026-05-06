def last_rate_limit(self):
        """
        A `dict` of the rate limit information returned in the most recent
        response, or `None` if no requests have been made yet.  The `dict`
        consists of all headers whose names begin with ``"RateLimit"`` (case
        insensitive).

        The DigitalOcean API specifies the following rate limit headers:

        :var string RateLimit-Limit: the number of requests that can be made
            per hour
        :var string RateLimit-Remaining: the number of requests remaining until
            the limit is reached
        :var string RateLimit-Reset: the Unix timestamp for the time when the
            oldest request will expire from rate limit consideration
        """
        if self.last_response is None:
            return None
        else:
            return {k:v for k,v in iteritems(self.last_response.headers)
                        if k.lower().startswith('ratelimit')}