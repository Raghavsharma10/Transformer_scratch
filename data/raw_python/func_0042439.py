def find(self, header, list_type=None):
        """Find the first chunk with specified header and optional list type."""
        for chunk in self:
            if chunk.header == header and (list_type is None or (header in
                    list_headers and chunk.type == list_type)):
                return chunk
            elif chunk.header in list_headers:
                try:
                    result = chunk.find(header, list_type)
                    return result
                except chunk.NotFound:
                    pass
        if list_type is None:
            raise self.NotFound('Chunk \'{0}\' not found.'.format(header))
        else:
            raise self.NotFound('List \'{0} {1}\' not found.'.format(header,
                list_type))