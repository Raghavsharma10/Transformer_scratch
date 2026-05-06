def get_headers(self, action, headers_ext=None):
        """Returns HTTP headers of specified WebDAV actions.

        :param action: the identifier of action.
        :param headers_ext: (optional) the addition headers list witch sgould be added to basic HTTP headers for
                            the specified action.
        :return: the dictionary of headers for specified action.
        """
        if action in Client.http_header:
            try:
                headers = Client.http_header[action].copy()
            except AttributeError:
                headers = Client.http_header[action][:]
        else:
            headers = list()

        if headers_ext:
            headers.extend(headers_ext)

        if self.webdav.token:
            webdav_token = f'Authorization: OAuth {self.webdav.token}'
            headers.append(webdav_token)
        return dict([map(lambda s: s.strip(), i.split(':')) for i in headers])