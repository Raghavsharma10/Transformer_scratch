def _get_content_type(url):
        """Get the Content-Type of the given url, using a HEAD request"""
        scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
        if not scheme in ('http', 'https', 'ftp', 'ftps'):
            ## FIXME: some warning or something?
            ## assertion error?
            return ''
        req = Urllib2HeadRequest(url, headers={'Host': netloc})
        resp = urlopen(req)
        try:
            if hasattr(resp, 'code') and resp.code != 200 and scheme not in ('ftp', 'ftps'):
                ## FIXME: doesn't handle redirects
                return ''
            return resp.info().get('content-type', '')
        finally:
            resp.close()