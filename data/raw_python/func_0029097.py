def xml(self, url, method='get', params=None, data=None):
        """
        请求并返回xml
        
        :type url: str
        :param url: API
        
        :type method: str
        :param method: HTTP METHOD
        
        :type params: dict
        :param params: query
        
        :type data: dict
        :param data: body
        
        :rtype: html.HtmlElement
        :return: 
        """
        r = self.req(url, method, params, data)
        # this is required for avoid utf8-mb4 lead to encoding error
        return self.to_xml(r.content, base_url=r.url)