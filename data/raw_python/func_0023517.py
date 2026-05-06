def parse_info_response(content, path, hostname):
        """Parses of response content XML from WebDAV server and extract an information about resource.

        :param content: the XML content of HTTP response from WebDAV server.
        :param path: the path to resource.
        :param hostname: the server hostname.
        :return: a dictionary of information attributes and them values with following keys:
                 `created`: date of resource creation,
                 `name`: name of resource,
                 `size`: size of resource,
                 `modified`: date of resource modification.
        """
        response = WebDavXmlUtils.extract_response_for_path(content=content, path=path, hostname=hostname)
        find_attributes = {
            'created': './/{DAV:}creationdate',
            'name': './/{DAV:}displayname',
            'size': './/{DAV:}getcontentlength',
            'modified': './/{DAV:}getlastmodified'
        }
        info = dict()
        for (name, value) in find_attributes.items():
            info[name] = response.findtext(value)
        return info