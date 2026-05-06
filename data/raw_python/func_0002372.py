def _create_path(self, *args):
        """Create the URL path with the Fred endpoint and given arguments."""
        args = filter(None, args)
        path = self.endpoint + '/'.join(args)
        return path