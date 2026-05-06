def _parse_xml(self, response):
        """
        Run our XML parser (lxml in this case) over our response text.  Lxml
        doesn't enjoy having xml/encoding information in the header so we strip
        that out if necessary. We return a parsed XML object that can be
        used by the calling API method and massaged into a more appropriate
        format.
        """
        if response.startswith('\n'):
            response = response[1:]
        tree = etree.fromstring(response)
        return tree