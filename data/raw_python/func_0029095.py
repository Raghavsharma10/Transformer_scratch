def req(self, url, method='get', params=None, data=None, auth=False):
        """
        请求API

        :type url: str
        :param url: API
        
        :type method: str
        :param method: HTTP METHOD
        
        :type params: dict
        :param params: query
        
        :type data: dict
        :param data: body
        
        :type auth: bool
        :param auth: if True and session expired will raise exception
        
        :rtype: requests.Response
        :return: Response
        """
        self.logger.debug('fetch api<%s:%s>' % (method, url))
        if auth and self.user_alias is None:
            raise Exception('cannot fetch api<%s> without session' % url)
        s = requests.Session()
        r = s.request(method, url, params=params, data=data, cookies=self.cookies, headers=self.headers,
                      timeout=self.timeout)
        s.close()
        if r.url is not url and RE_SESSION_EXPIRE.search(r.url) is not None:
            self.expire()
            if auth:
                raise Exception('auth expired, could not fetch with<%s>' % url)
        return r