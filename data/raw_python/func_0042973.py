def get_api_link(self):
        """
        Adds a query string to the api url. At minimum adds the type=choices
        argument so that the return format is json. Any other filtering
        arguments calculated by the `get_qs` method are then added to the
        url. It is up to the destination url to respect them as filters.
        """
        url = self._api_link
        if url:
            qs = self.get_qs()
            url = "%s?type=choices" % url
            if qs:
                url = "%s&amp;%s" % (url, u'&amp;'.join([u'%s=%s' % (k, urllib.quote(unicode(v).encode('utf8'))) \
                                                        for k, v in qs.items()]))
                url = "%s&amp;%s" % (url, u'&amp;'.join([u'exclude=%s' % x \
                                                        for x in qs.keys()]))
        return url