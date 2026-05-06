def find_all(self, header, list_type=None):
        """Find all direct children with header and optional list type."""
        found = []
        for chunk in self:
            if chunk.header == header and (not list_type or (header in
                list_headers and chunk.type == list_type)):
                found.append(chunk)
        return found