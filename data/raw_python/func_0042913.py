def request(self, url, post=None, method="GET"):
        """ Make the request"""
        dsid = self.get_dsid()
        baseurl = "https://auth.api.swedbank.se/TDE_DAP_Portal_REST_WEB/api/v1/%s?dsid=%s" % (
            url, dsid)

        if self.pch is None:
            self.pch = build_opener(HTTPCookieProcessor(self.cj))

        if post:
            post = bytearray(post, "utf-8")
            request = Request(baseurl, data=post)
            request.add_header("Content-Type", "application/json")
        else:
            request = Request(baseurl)

        request.add_header("User-Agent", self.useragent)
        request.add_header("Authorization", self.get_authkey())
        request.add_header("Accept", "*/*")
        request.add_header("Accept-Language", "sv-se")
        request.add_header("Connection", "keep-alive")
        request.add_header("Proxy-Connection", "keep-alive")
        self.cj.set_cookie(
                Cookie(version=0, name='dsid', value=dsid, port=None,
                       port_specified=False, domain='.api.swedbank.se',
                       domain_specified=False, domain_initial_dot=False,
                       path='/',
                       path_specified=True, secure=False, expires=None,
                       discard=True, comment=None, comment_url=None,
                       rest={'HttpsOnly': None}, rfc2109=False))
        request.get_method = lambda: method
        tmp = self.pch.open(request)
        self.data = tmp.read().decode("utf8")