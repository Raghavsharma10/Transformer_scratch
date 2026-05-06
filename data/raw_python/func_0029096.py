def json(self, url, method='get', params=None, data=None):
        """
        请求并返回json
        
        
        :type url: str
        :param url: API
        
        :type method: str
        :param method: HTTP METHOD
        
        :type params: dict
        :param params: query
        
        :type data: dict
        :param data: body
        
        :rtype: dict
        :return: 
        """
        r = self.req(url, method, params, data)
        return r.json()