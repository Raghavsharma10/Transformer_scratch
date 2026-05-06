def parse_get_list_response(content):
        """Parses of response content XML from WebDAV server and extract file and directory names.

        :param content: the XML content of HTTP response from WebDAV server for getting list of files by remote path.
        :return: list of extracted file or directory names.
        """
        try:
            tree = etree.fromstring(content)
            hrees = [Urn.separate + unquote(urlsplit(hree.text).path) for hree in tree.findall('.//{DAV:}href')]
            return [Urn(hree) for hree in hrees]
        except etree.XMLSyntaxError:
            return list()