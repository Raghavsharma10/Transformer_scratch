def parse_is_dir_response(content, path, hostname):
        """Parses of response content XML from WebDAV server and extract an information about resource.

        :param content: the XML content of HTTP response from WebDAV server.
        :param path: the path to resource.
        :param hostname: the server hostname.
        :return: True in case the remote resource is directory and False otherwise.
        """
        response = WebDavXmlUtils.extract_response_for_path(content=content, path=path, hostname=hostname)
        resource_type = response.find('.//{DAV:}resourcetype')

        if resource_type is None:
            raise MethodNotSupported(name='is_dir', server=hostname)

        dir_type = resource_type.find('{DAV:}collection')

        return dir_type is not None