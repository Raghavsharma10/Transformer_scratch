def get_backend_name(self, location):
        """
        Return the name of the version control backend if found at given
        location, e.g. vcs.get_backend_name('/path/to/vcs/checkout')
        """
        for vc_type in self._registry.values():
            path = os.path.join(location, vc_type.dirname)
            if os.path.exists(path):
                return vc_type.name
        return None