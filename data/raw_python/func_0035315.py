def send(self):
        """ Post fields and files to an HTTP server as multipart/form-data.
            Return the server's response.
        """
        scheme, location, path, query, _ = urlparse.urlsplit(self.url)
        assert scheme in ("http", "https"), "Unsupported scheme %r" % scheme

        content_type, body = self._encode_multipart_formdata()
        handle = getattr(httplib, scheme.upper() + "Connection")(location)
        if self.mock_http:
            # Don't actually send anything, print to stdout instead
            handle.sock = parts.Bunch(
                sendall=lambda x: sys.stdout.write(fmt.to_utf8(
                    ''.join((c if 32 <= ord(c) < 127 or ord(c) in (8, 10) else u'\u27ea%02X\u27eb' % ord(c)) for c in x)
                )),
                makefile=lambda dummy, _: StringIO.StringIO("\r\n".join((
                    "HTTP/1.0 204 NO CONTENT",
                    "Content-Length: 0",
                    "",
                ))),
                close=lambda: None,
            )

        handle.putrequest('POST', urlparse.urlunsplit(('', '', path, query, '')))
        handle.putheader('Content-Type', content_type)
        handle.putheader('Content-Length', str(len(body)))
        for key, val in self.headers.items():
            handle.putheader(key, val)
        handle.endheaders()
        handle.send(body)
        #print handle.__dict__

        return handle.getresponse()