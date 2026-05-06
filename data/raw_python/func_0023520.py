def parse_get_property_response(content, name):
        """Parses of response content XML from WebDAV server for getting metadata property value for some resource.

        :param content: the XML content of response as string.
        :param name: the name of property for finding a value in response
        :return: the value of property if it has been found or None otherwise.
        """
        tree = etree.fromstring(content)
        return tree.xpath('//*[local-name() = $name]', name=name)[0].text