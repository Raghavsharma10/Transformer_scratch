def is_private(self, key, sources):
        """Check if attribute is private."""
        # aliases are always public.
        if key == ENTRY.ALIAS:
            return False
        return all([
            SOURCE.PRIVATE in self.get_source_by_alias(x)
            for x in sources.split(',')
        ])