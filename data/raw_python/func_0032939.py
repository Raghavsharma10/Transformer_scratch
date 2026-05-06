def _begin_request(self):
        """
        Actually start executing this request.
        """
        headers = self.m2req.headers

        self._request = HTTPRequest(connection=self,
            method=headers.get("METHOD"),
            uri=self.m2req.path,
            version=headers.get("VERSION"),
            headers=headers,
            remote_ip=headers.get("x-forwarded-for"))

        if len(self.m2req.body) > 0:
            self._request.body = self.m2req.body

        if self.m2req.is_disconnect():
            self.finish()

        elif headers.get("x-mongrel2-upload-done", None):
            # there has been a file upload.
            expected = headers.get("x-mongrel2-upload-start", "BAD")
            upload = headers.get("x-mongrel2-upload-done", None)

            if expected == upload:
                self.request_callback(self._request)

        elif headers.get("x-mongrel2-upload-start", None):
            # this is just a notification that a file upload has started. Do
            # nothing for now!
            pass

        else:
            self.request_callback(self._request)