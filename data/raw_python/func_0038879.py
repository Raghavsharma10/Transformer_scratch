def _contents(self):
        """The raw contents of the URL as fetched, this is done lazily.
           For non-lazy fetching this is accessed in the object constructor."""
        if self.__urldata__ is Ellipsis or self.__cache_request__ is False:
            if self._file_data:
                # Special-case: do a multipart upload if there's file data
                self.__post__ = True
                boundary = "-"*12+str(uuid.uuid4())+"$"
                multipart_data = ''
                for k, v in cgi.parse_qs(self.query).items():
                    if not isinstance(v, list):
                        v = [v]
                    for val in v:
                        multipart_data += boundary + "\r\n"
                        multipart_data += ('Content-Disposition: form-data; '
                                           'name="%s"\r\n\r\n' % k)
                        multipart_data += val + "\r\n"
                for k, v in self._file_data.items():
                    fn = os.path.basename(getattr(v, 'name', 'file'))
                    ct = (mimetypes.guess_type(fn) 
                            or ("application/octet-stream",))[0]
                    multipart_data += boundary + "\r\n"
                    multipart_data += ('Content-Disposition: form-data; '
                                       'name="%s"; filename="%s"\r\n'
                                       'Content-Type:%s\r\n\r\n' % 
                                            (k, fn, ct))
                    multipart_data += v.read() + "\r\n"
                multipart_data += boundary + "--\r\n\r\n"
                req_dict = {'User-Agent' : USER_AGENT,
                            'Content-Type': 
                                'multipart/form-data; boundary='+boundary[2:],
                            'Content-Length': str(len(multipart_data))
                            }
                if self._referer:
                    req_dict['Referer'] = self._referer
                request = compat.urllib2.Request(self.url,
                                          multipart_data,
                                          req_dict)
            else:
                req_dict = {'User-Agent' : USER_AGENT}
                if self._referer:
                    req_dict['Referer'] = self._referer
                request = compat.urllib2.Request(self.url, self.query 
                                                        if self.__post__
                                                        else None,
                                           req_dict)
            handle = compat.urllib2.urlopen(request)
            # Handle the special case of a redirect (only follow once) --
            # Note that only the first 3 components (protocol, hostname, path)
            # are altered as component 4 is the query string, which can get
            # clobbered by the server.
            fetched_url = list(compat.urlsplit(handle.url)[:3])
            if fetched_url != list(self._url[:3]):
                self._url[:3] = fetched_url
                return self._contents
            # No redirect, proceed as usual.
            self.__headers__ = compat.get_headers(handle)
            self.__urldata__ = handle.read()
        data = self.__urldata__
        if self.__cache_request__ is False:
            self.__urldata__ = Ellipsis
        return data