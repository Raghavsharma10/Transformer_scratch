def request(self, url, method="GET", body=None, headers=None, redirections=DEFAULT_MAX_REDIRECTS, connection_type=None, blocking=True):
        """
        Make a network request by calling QgsNetworkAccessManager.
        redirections argument is ignored and is here only for httplib2 compatibility.
        """
        self.msg_log(u'http_call request: {0}'.format(url))

        self.blocking_mode = blocking
        req = QNetworkRequest()
        # Avoid double quoting form QUrl
        url = urllib.parse.unquote(url)
        req.setUrl(QUrl(url))
        if headers is not None:
            # This fixes a wierd error with compressed content not being correctly
            # inflated.
            # If you set the header on the QNetworkRequest you are basically telling
            # QNetworkAccessManager "I know what I'm doing, please don't do any content
            # encoding processing".
            # See: https://bugs.webkit.org/show_bug.cgi?id=63696#c1
            try:
                del headers['Accept-Encoding']
            except KeyError:
                pass
            for k, v in list(headers.items()):
                self.msg_log("Setting header %s to %s" % (k, v))
                req.setRawHeader(k.encode(), v.encode())
        if self.authid:
            self.msg_log("Update request w/ authid: {0}".format(self.authid))
            self.auth_manager().updateNetworkRequest(req, self.authid)
        if self.reply is not None and self.reply.isRunning():
            self.reply.close()
        if method.lower() == 'delete':
            func = getattr(QgsNetworkAccessManager.instance(), 'deleteResource')
        else:
            func = getattr(QgsNetworkAccessManager.instance(), method.lower())
        # Calling the server ...
        # Let's log the whole call for debugging purposes:
        self.msg_log("Sending %s request to %s" % (method.upper(), req.url().toString()))
        self.on_abort = False
        headers = {str(h): str(req.rawHeader(h)) for h in req.rawHeaderList()}
        for k, v in list(headers.items()):
            self.msg_log("%s: %s" % (k, v))
        if method.lower() in ['post', 'put']:
            if isinstance(body, io.IOBase):
                body = body.read()
            if isinstance(body, str):
                body = body.encode()
            self.reply = func(req, body)
        else:
            self.reply = func(req)
        if self.authid:
            self.msg_log("Update reply w/ authid: {0}".format(self.authid))
            self.auth_manager().updateNetworkReply(self.reply, self.authid)

        # necessary to trap local timout manage by QgsNetworkAccessManager
        # calling QgsNetworkAccessManager::abortRequest
        QgsNetworkAccessManager.instance().requestTimedOut.connect(self.requestTimedOut)

        self.reply.sslErrors.connect(self.sslErrors)
        self.reply.finished.connect(self.replyFinished)
        self.reply.downloadProgress.connect(self.downloadProgress)

        # block if blocking mode otherwise return immediatly
        # it's up to the caller to manage listeners in case of no blocking mode
        if not self.blocking_mode:
            return (None, None)

        # Call and block
        self.el = QEventLoop()
        self.reply.finished.connect(self.el.quit)

        # Catch all exceptions (and clean up requests)
        try:
            self.el.exec_(QEventLoop.ExcludeUserInputEvents)
        except Exception as e:
            raise e

        if self.reply:
            self.reply.finished.disconnect(self.el.quit)

        # emit exception in case of error
        if not self.http_call_result.ok:
            if self.http_call_result.exception and not self.exception_class:
                raise self.http_call_result.exception
            else:
                raise self.exception_class(self.http_call_result.reason)

        return (self.http_call_result, self.http_call_result.content)