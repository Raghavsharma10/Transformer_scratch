def _encode_multipart_formdata(self):
        """ Encode POST body.
            Return (content_type, body) ready for httplib.HTTP instance
        """
        def get_content_type(filename):
            "Helper to get MIME type."
            return mimetypes.guess_type(filename)[0] or 'application/octet-stream'

        boundary = '----------ThIs_Is_tHe_b0uNdaRY_%d$' % (time.time())
        logical_lines = []
        for name, value in self.fields:
            if value is None:
                continue
            logical_lines.append('--' + boundary)
            if hasattr(value, "read"):
                filename = getattr(value, "name", str(id(value))+".dat")
                logical_lines.append('Content-Disposition: form-data; name="%s"; filename="%s"' % (
                    name,
                    os.path.basename(filename).replace("'", '_').replace('"', '_')
                ))
                logical_lines.append('Content-Type: %s' % get_content_type(filename))
                logical_lines.append('Content-Transfer-Encoding: binary')
                value = value.read()
            else:
                logical_lines.append('Content-Disposition: form-data; name="%s"' % name)
                logical_lines.append('Content-Type: text/plain; charset="UTF-8"')
                value = fmt.to_utf8(value)
            #logical_lines.append('Content-Length: %d' % len(value))
            logical_lines.append('')
            logical_lines.append(value)
        logical_lines.append('--' + boundary + '--')
        logical_lines.append('')

        body = '\r\n'.join(logical_lines)
        content_type = 'multipart/form-data; boundary=%s' % boundary
        return content_type, body