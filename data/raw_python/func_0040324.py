def _request(self, url, method = u"get", data = None, headers=None, **kwargs):
        """
        does the request via requests
        - oauth not implemented yet
        - use basic auth please
        """
        #        if self.access_token:
        #            auth_header = {
        #                u"Authorization": "Bearer %s" % (self.access_token)
        #            }
        #            headers.update(auth_header)
        #basic auth
        msg = "method: %s url:%s\nheaders:%s\ndata:%s" % (
            method, url, headers, data)
        #print msg
        if not self.use_oauth:
            auth = (self.sk_user, self.sk_pw)
            if not self.client:
                self.client = requests.session()
            r = self.client.request(method, url, headers=headers, data=data, auth=auth,**kwargs)
        else:
            if not self.client:
                self.client = requests.session(hooks={'pre_request': oauth_hook})
            r = self.client.request(method, url, headers=headers, data=data,**kwargs)
        return r