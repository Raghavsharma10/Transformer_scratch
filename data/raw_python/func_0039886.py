def compare(self, path, prefixed_path, source_storage):
        """
        Returns True if the file should be copied.
        """
        # First try a method on the command named compare_<comparison_method>
        # If that doesn't exist, create a comparitor that calls methods on the
        # storage with the name <comparison_method>, passing them the name.
        comparitor = getattr(self, 'compare_%s' % self.comparison_method, None)
        if not comparitor:
            comparitor = self._create_comparitor(self.comparison_method)
        return comparitor(path, prefixed_path, source_storage)