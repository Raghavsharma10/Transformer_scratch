def post_operations(self, mode=None):
        """ Return post-operations only for the mode asked """
        version_mode = self._get_version_mode(mode=mode)
        return version_mode.post_operations