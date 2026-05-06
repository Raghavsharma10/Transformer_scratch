def getMeta(self, uri):
        """Return meta information about an action. Cache the result as specified by the server"""

        action = urlparse(uri).path

        mediaKey = self.cacheKey + '_meta_' + action
        mediaKey = mediaKey.replace(' ', '__')

        meta = cache.get(mediaKey, None)

        # Nothing found -> Retrieve it from the server and cache it
        if not meta:

            r = self.doQuery('meta/' + uri)

            if r.status_code == 200:  # Get the content if there is not problem. If there is, template will stay to None
                meta = r.json()

            if 'expire' not in r.headers:
                expire = 5 * 60  # 5 minutes of cache if the server didn't specified anything
            else:
                expire = int((parser.parse(r.headers['expire']) - datetime.datetime.now(tzutc())).total_seconds())  # Use the server value for cache

            if expire > 0:  # Do the server want us to cache ?
                cache.set(mediaKey, meta, expire)

        return meta