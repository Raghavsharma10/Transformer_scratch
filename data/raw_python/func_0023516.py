def parse_free_space_response(content, hostname):
        """Parses of response content XML from WebDAV server and extract an amount of free space.

        :param content: the XML content of HTTP response from WebDAV server for getting free space.
        :param hostname: the server hostname.
        :return: an amount of free space in bytes.
        """
        try:
            tree = etree.fromstring(content)
            node = tree.find('.//{DAV:}quota-available-bytes')
            if node is not None:
                return int(node.text)
            else:
                raise MethodNotSupported(name='free', server=hostname)
        except TypeError:
            raise MethodNotSupported(name='free', server=hostname)
        except etree.XMLSyntaxError:
            return str()