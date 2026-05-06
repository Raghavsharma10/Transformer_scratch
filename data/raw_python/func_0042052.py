def get_add_link(self):
        """
        Appends the popup=1 query string to the url so the
        destination url treats it as a popup.
        """

        url = super(TaggedRelationWidget, self).get_add_link()
        if url:
            qs = self.get_add_qs()
            if qs:
                url = "%s&%s" % (url, urllib.urlencode(qs))
        return url