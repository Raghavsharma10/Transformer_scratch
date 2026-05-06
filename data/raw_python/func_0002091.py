def set(self, parent_url, fp):
        """
        If the parent object already has XML metadata, it will be overwritten.

        Accepts XML metadata in any of the three supported formats.
        The format will be detected from the XML content.

        The Metadata object becomes invalid after setting

        :param file fp: A reference to an open file-like object which the content will be read from.
        """
        url = parent_url + self.client.get_url_path('METADATA', 'POST', 'set', {})
        r = self.client.request('POST', url, data=fp, headers={'Content-Type': 'text/xml'})
        if r.status_code not in [200, 201]:
            raise exceptions.ServerError("Expected success response, got %s: %s" % (r.status_code, url))