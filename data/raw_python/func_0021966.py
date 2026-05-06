def to_string(self, limit=None):
        """
        Create a string representation of this collection, showing up to
        `limit` items.
        """
        header = self.short_string()
        if len(self) == 0:
            return header
        contents = ""
        element_lines = [
            "  -- %s" % (element,)
            for element in self.elements[:limit]
        ]
        contents = "\n".join(element_lines)

        if limit is not None and len(self.elements) > limit:
            contents += "\n  ... and %d more" % (len(self) - limit)
        return "%s\n%s" % (header, contents)