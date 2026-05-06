def url(proto, server, port=None, uri=None):
        """Construct a URL from the given components."""
        url_parts = [proto, '://', server]
        if port:
            port = int(port)
            if port < 1 or port > 65535:
                raise ValueError('invalid port value')
            if not ((proto == 'http' and port == 80) or
                    (proto == 'https' and port == 443)):
                url_parts.append(':')
                url_parts.append(str(port))

        if uri:
            url_parts.append('/')
            url_parts.append(requests.utils.quote(uri.strip('/')))

        url_parts.append('/')
        return ''.join(url_parts)