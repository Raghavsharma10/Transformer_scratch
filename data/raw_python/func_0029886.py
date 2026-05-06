def enable_cache(self):
        """ Enable client-side caching for the current request """
        self.set_header('Cache-Control', 'max-age=%d, public' % self.CACHE_TIME)

        now = datetime.datetime.now()
        expires = now + datetime.timedelta(seconds=self.CACHE_TIME)
        self.set_header('Expires', expires.strftime('%a, %d %b %Y %H:%M:%S'))

        self.set_header('access-control-max-age', self.CACHE_TIME)