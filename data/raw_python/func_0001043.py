def fetch(self, url, encoding=None, force_refetch=False, nocache=False, quiet=True):
        ''' Fetch a HTML file as binary'''
        try:
            if not force_refetch and self.cache is not None and url in self.cache:
                # try to look for content in cache
                logging.debug('Retrieving content from cache for {}'.format(url))
                return self.cache.retrieve_blob(url, encoding)
            encoded_url = WebHelper.encode_url(url)
            req = Request(encoded_url, headers={'User-Agent': 'Mozilla/5.0'})
            # support gzip
            req.add_header('Accept-encoding', 'gzip, deflate')
            # Open URL
            getLogger().info("Fetching: {url} |".format(url=url))
            response = urlopen(req)
            content = response.read()
            # unzip if required
            if 'Content-Encoding' in response.info() and response.info().get('Content-Encoding') == 'gzip':
                # unzip
                with gzip.open(BytesIO(content)) as gzfile:
                    content = gzfile.read()
            # update cache if required
            if self.cache is not None and not nocache:
                if url not in self.cache:
                    self.cache.insert_blob(url, content)
            return content.decode(encoding) if content and encoding else content
        except URLError as e:
            if hasattr(e, 'reason'):
                getLogger().exception('We failed to reach {}. Reason: {}'.format(url, e.reason))
            elif hasattr(e, 'code'):
                getLogger().exception('The server couldn\'t fulfill the request. Error code: {}'.format(e.code))
            else:
                # Other exception ...
                getLogger().exception("Fetching error")
            if not quiet:
                raise
        return None