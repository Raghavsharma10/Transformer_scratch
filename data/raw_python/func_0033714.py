def doQuery(self, url, method='GET', getParmeters=None, postParameters=None, files=None, extraHeaders={}, session={}):
        """Send a request to the server and return the result"""

        # Build headers
        headers = {}

        if not postParameters:
            postParameters = {}

        for key, value in extraHeaders.iteritems():
            # Fixes #197 for values with utf-8 chars to be passed into plugit
            if isinstance(value, basestring):
                headers['X-Plugit-' + key] = value.encode('utf-8')
            else:
                headers['X-Plugit-' + key] = value

        for key, value in session.iteritems():
            headers['X-Plugitsession-' + key] = value
            if 'Cookie' not in headers:
                headers['Cookie'] = ''
            headers['Cookie'] += key + '=' + str(value) + '; '

        if method == 'POST':
            if not files:
                r = requests.post(self.baseURI + '/' + url, params=getParmeters, data=postParameters, stream=True, headers=headers)
            else:
                # Special way, for big files
                # Requests is not usable: https://github.com/shazow/urllib3/issues/51

                from poster.encode import multipart_encode, MultipartParam
                from poster.streaminghttp import register_openers
                import urllib2
                import urllib

                # Register the streaming http handlers with urllib2
                register_openers()

                # headers contains the necessary Content-Type and Content-Length
                # datagen is a generator object that yields the encoded parameters
                data = []
                for x in postParameters:
                    if isinstance(postParameters[x], list):
                        for elem in postParameters[x]:
                            data.append((x, elem))
                    else:
                        data.append((x, postParameters[x]))

                for f in files:
                    data.append((f, MultipartParam(f, fileobj=open(files[f].temporary_file_path(), 'rb'), filename=files[f].name)))

                datagen, headers_multi = multipart_encode(data)

                headers.update(headers_multi)

                if getParmeters:
                    get_uri = '?' + urllib.urlencode(getParmeters)
                else:
                    get_uri = ''

                # Create the Request object
                request = urllib2.Request(self.baseURI + '/' + url + get_uri, datagen, headers)

                re = urllib2.urlopen(request)

                from requests import Response

                r = Response()
                r.status_code = re.getcode()
                r.headers = dict(re.info())
                r.encoding = "application/json"
                r.raw = re.read()
                r._content = r.raw

                return r

        else:
            # Call the function based on the method.
            r = requests.request(method.upper(), self.baseURI + '/' + url, params=getParmeters, stream=True, headers=headers, allow_redirects=True)

        return r