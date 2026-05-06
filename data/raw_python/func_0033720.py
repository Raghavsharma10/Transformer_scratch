def getTemplate(self, uri, meta=None):
        """Return the template for an action. Cache the result. Can use an optional meta parameter with meta information"""

        if not meta:

            metaKey = self.cacheKey + '_templatesmeta_cache_' + uri

            meta = cache.get(metaKey, None)

            if not meta:
                meta = self.getMeta(uri)
                cache.set(metaKey, meta, 15)

        if not meta:  # No meta, can return a template
            return None

        # Let's find the template in the cache
        action = urlparse(uri).path

        templateKey = self.cacheKey + '_templates_' + action + '_' + meta['template_tag']
        template = cache.get(templateKey, None)

        # Nothing found -> Retrieve it from the server and cache it
        if not template:

            r = self.doQuery('template/' + uri)

            if r.status_code == 200:  # Get the content if there is not problem. If there is, template will stay to None
                template = r.content

            cache.set(templateKey, template, None)  # None = Cache forever

        return template