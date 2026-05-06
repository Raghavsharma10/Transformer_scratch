async def rewrite(self, path, method = None, keepresponse = True):
        "Rewrite this request to another processor. Must be called before header sent"
        if self._sendHeaders:
            raise HttpProtocolException('Cannot modify response, headers already sent')
        if getattr(self.event, 'rewritedepth', 0) >= getattr(self.protocol, 'rewritedepthlimit', 32):
            raise HttpRewriteLoopException
        newpath = urljoin(quote_from_bytes(self.path).encode('ascii'), path)
        if newpath == self.fullpath or newpath == self.originalpath:
            raise HttpRewriteLoopException
        extraparams = {}
        if keepresponse:
            if hasattr(self, 'status'):
                extraparams['status'] = self.status
            extraparams['sent_headers'] = self.sent_headers
            extraparams['sent_cookies'] = self.sent_cookies
        r = HttpRequestEvent(self.host,
                               newpath,
                               self.method if method is None else method,
                               self.connection,
                               self.connmark,
                               self.xid,
                               self.protocol,
                               headers = self.headers,
                               headerdict = self.headerdict,
                               setcookies = self.setcookies,
                               stream = self.inputstream,
                               rewritefrom = self.fullpath,
                               originalpath = self.originalpath,
                               rewritedepth = getattr(self.event, 'rewritedepth', 0) + 1,
                               **extraparams
                               )
        await self.connection.wait_for_send(r)
        self._sendHeaders = True
        self.outputstream = None