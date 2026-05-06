def build_opener(self):
        """
        Builds url opener, initializing proxy.
        @return: OpenerDirector
        """
        http_handler = urllib2.HTTPHandler() # debuglevel=self.transport.debug

        if util.empty(self.transport.proxy_url):
            return urllib2.build_opener(http_handler)

        proxy_handler = urllib2.ProxyHandler(
            {self.transport.proxy_url[:4]: self.transport.proxy_url})

        return urllib2.build_opener(http_handler, proxy_handler)