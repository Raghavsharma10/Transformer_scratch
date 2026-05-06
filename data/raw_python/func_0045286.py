def req(self, meth, url, http_data=''):
        """
        sugar that wraps the 'requests' module with basic auth and some headers.
        """
        self.logger.debug("Making request: %s %s\nBody:%s" % (meth, url, http_data))
        req_method = getattr(requests, meth)
        return (req_method(url,
                           auth=(self.__username, self.__password),
                           data=http_data,
                           headers=({'user-agent': self.user_agent(), 'Accept': 'application/json'})))