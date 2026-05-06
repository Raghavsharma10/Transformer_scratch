def get_binary(self):
        """Return a binary buffer containing the file content"""

        content_disp = 'Content-Disposition: form-data; name="file"; filename="{}"'

        stream = io.BytesIO()
        stream.write(_string_to_binary('--{}'.format(self.boundary)))
        stream.write(_crlf())
        stream.write(_string_to_binary(content_disp.format(self.file_name)))
        stream.write(_crlf())
        stream.write(_crlf())
        stream.write(self.body)
        stream.write(_crlf())
        stream.write(_string_to_binary('--{}--'.format(self.boundary)))
        stream.write(_crlf())

        return stream.getvalue()