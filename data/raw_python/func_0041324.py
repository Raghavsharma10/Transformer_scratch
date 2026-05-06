def pack_bytes(self, obj_dict, encoding=None):
        """Pack a dictionary into a byte stream."""
        assert self.dict_to_bytes or self.dict_to_string
        encoding = encoding or self.default_encoding or 'utf-8'
        LOGGER.debug('%r encoding dict with encoding %s', self, encoding)
        if self.dict_to_bytes:
            return None, self.dict_to_bytes(obj_dict)
        try:
            return encoding, self.dict_to_string(obj_dict).encode(encoding)
        except LookupError as error:
            raise web.HTTPError(
                406, 'failed to encode result %r', error,
                reason='target charset {0} not found'.format(encoding))
        except UnicodeEncodeError as error:
            LOGGER.warning('failed to encode text as %s - %s, trying utf-8',
                           encoding, str(error))
            return 'utf-8', self.dict_to_string(obj_dict).encode('utf-8')