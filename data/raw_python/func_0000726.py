def prepare(self, mime):
        """
        Prepares a MIME object by applying the headers
        to the *mime* object. Ignores any Bcc or
        Resent-Bcc headers.
        """
        for key in self:
            if key == 'Bcc' or key == 'Resent-Bcc':
                continue
            del mime[key]
            # Python 3.* email's compatibility layer will handle
            # unicode field values in proper way but Python 2
            # won't (it will encode not only additional field
            # values but also all header values)
            parsed_header, additional_fields = parse_header(
                self[key] if IS_PY3 else
                self[key].encode("utf-8")
            )
            mime.add_header(key, parsed_header, **additional_fields)