def should_expand(self, tag):
        """Return whether the specified tag should be expanded."""
        return self.indentation is not None and tag and (
            not self.previous_indent or (
                tag.serializer == 'list'
                and tag.subtype.serializer in ('array', 'list', 'compound')
            ) or (
                tag.serializer == 'compound'
            )
        )