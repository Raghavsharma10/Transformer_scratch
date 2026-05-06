def getMedia(self, uri):
        """Return a tuple with a media and his content-type. Don't cache anything !"""

        r = self.doQuery('media/' + uri)

        if r.status_code == 200:
            content_type = 'application/octet-stream'

            if 'content-type' in r.headers:
                content_type = r.headers['content-type']

            cache_control = None

            if 'cache-control' in r.headers:
                cache_control = r.headers['cache-control']

            return (r.content, content_type, cache_control)
        else:
            return (None, None, None)