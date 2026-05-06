def base_path(self):
        """
        Calculate the APIs base path
        """
        path = UrlPath()

        # Walk up the API to find the base object
        parent = self.parent
        while parent:
            path_prefix = getattr(parent, 'path_prefix', NoPath)
            path = path_prefix + path
            parent = getattr(parent, 'parent', None)

        return path