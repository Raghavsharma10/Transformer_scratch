def unpack_bytes(self, obj_bytes, encoding=None):
        """Unpack a byte stream into a dictionary."""
        assert self.bytes_to_dict or self.string_to_dict
        encoding = encoding or self.default_encoding
        LOGGER.debug('%r decoding %d bytes with encoding of %s',
                     self, len(obj_bytes), encoding)
        if self.bytes_to_dict:
            return escape.recursive_unicode(self.bytes_to_dict(obj_bytes))
        return self.string_to_dict(obj_bytes.decode(encoding))