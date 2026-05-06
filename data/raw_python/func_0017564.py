def _get_multiparts(response):
        """
        From this
        'multipart/parallel; boundary="874e43d27ec6d83f30f37841bdaf90c7"; charset=utf-8'
        get this
        --874e43d27ec6d83f30f37841bdaf90c7
        """
        boundary = None
        for part in response.headers.get('Content-Type', '').split(';'):
            if 'boundary=' in part:
                boundary = '--{}'.format(part.split('=', 1)[1].strip('\"'))
                break

        if not boundary:
            raise ParseError("Was not able to find the boundary between objects in a multipart response")

        if response.content is None:
            return []

        response_string = response.content

        if six.PY3:
            # Python3 returns bytes, decode for string operations
            response_string = response_string.decode('latin-1')

        #  help bad responses be more multipart compliant
        whole_body = response_string.strip('\r\n')
        no_front_boundary = whole_body.strip(boundary)
        # The boundary comes with some characters

        multi_parts = []
        for part in no_front_boundary.split(boundary):
            multi_parts.append(part.strip('\r\n'))

        return multi_parts