def column(self, source_header_or_pos):
        """
        Return a column by name or position

        :param source_header_or_pos: If a string, a source header name. If an integer, column position
        :return:
        """
        for c in self.columns:
            if c.source_header == source_header_or_pos:
                assert c.st_vid == self.vid
                return c
            elif c.position == source_header_or_pos:
                assert c.st_vid == self.vid
                return c

        else:
            return None