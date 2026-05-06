def ratelimits(self):
        """ :returns: Rate Limit headers
            :rtype: dict
        """
        # can't use a dict comprehension because we want python2.6 support
        r = {}
        keys = filter(lambda x: x.startswith("x-ratelimit-"), self.headers.keys())
        for key in keys:
            r[key.replace("x-ratelimit-", "")] = int(self.headers[key])
        return r