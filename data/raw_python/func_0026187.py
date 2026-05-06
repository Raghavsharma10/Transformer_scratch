def _match_depth(self, sect, depth):
        """
        Given a section and a depth level, walk back through the sections
        parents to see if the depth level matches a previous section.

        Return a reference to the right section,
        or raise a SyntaxError.
        """
        while depth < sect.depth:
            if sect is sect.parent:
                # we've reached the top level already
                raise SyntaxError()
            sect = sect.parent
        if sect.depth == depth:
            return sect
        # shouldn't get here
        raise SyntaxError()