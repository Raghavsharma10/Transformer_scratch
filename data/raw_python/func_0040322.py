def _pre_request(self, url, method = u"get", data = None, headers=None, **kwargs):
        """
        hook for manipulating the _pre request data
        """
        return (url, method, data, headers, kwargs)