def is_excluded(self, path, exclude=None):
        """
        Return if path is in exclude pattern.
        """
        for pattern in (exclude or self.exclude_pattern):
            if path.match(pattern):
                return True
        else:
            return False