def free(self):
        """Returns an amount of free space on remote WebDAV server.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_PROPFIND

        :return: an amount of free space in bytes.
        """
        data = WebDavXmlUtils.create_free_space_request_content()
        response = self.execute_request(action='free', path='', data=data)
        return WebDavXmlUtils.parse_free_space_response(response.content, self.webdav.hostname)