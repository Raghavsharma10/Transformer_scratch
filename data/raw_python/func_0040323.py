def _pre_request(self, url, method = u"get", data = None, headers=None, **kwargs):
        """
        hook for manipulating the _pre request data
        """
        header = {
            u"Content-Type": u"application/json",
            u"User-Agent": u"salesking_api_py_v1",
        }
        if headers:
            headers.update(header)
        else:
            headers = header
        if url.find(self.base_url) !=0:
            url = u"%s%s" %(self.base_url, url)
        return (url, method, data, headers, kwargs)