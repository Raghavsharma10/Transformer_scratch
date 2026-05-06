def _is_exempt(self, environ):
        """
        Returns True if this request's URL starts with one of the
        excluded paths.
        """
        exemptions = self.exclude_paths

        if exemptions:
            path = environ.get('PATH_INFO')
            for excluded_p in self.exclude_paths:
                if path.startswith(excluded_p):
                    return True

        return False