def get_headers(self, cmd, msg_length, extra_headers):
        """Returns the headers string based on command to execute"""
        cmd_header = "%s %s" % (cmd, PROTOCOL_VERSION)
        len_header = "Content-length: %s" % msg_length
        headers = [cmd_header, len_header]
        if self.user:
            user_header = "User: %s" % self.user
            headers.append(user_header)
        if self.gzip:
            headers.append("Compress: zlib")
        if extra_headers is not None:
            for key in extra_headers:
                if key.lower() != 'content-length':
                    headers.append("%s: %s" % (key, extra_headers[key]))
        headers.append('')
        headers.append('')
        return '\r\n'.join(headers)