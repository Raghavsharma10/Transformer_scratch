def search(self, 
                response_format = None, 
                key = None, 
                **kwargs):
        """
        Calls the API and returns a dictionary of the search results
        
        :param response_format: the format that the API uses for its response, 
                                includes JSON (.json) and JSONP (.jsonp). 
                                Defaults to '.json'.
                                
        :param key: a developer key. Defaults to key given when the articleAPI class was initialized.
        
        """
        if response_format is None:
            response_format = self.response_format
        if key is None:
            key = self.key
        
        url = '%s%s?%sapi-key=%s' % (
            API_ROOT, response_format, self._options(**kwargs), key
        )
        
        r = requests.get(url)
        return r.json()