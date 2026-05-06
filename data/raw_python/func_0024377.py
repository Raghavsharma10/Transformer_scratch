def pre_operations(self, mode=None):
        """ Return pre-operations only for the mode asked """
        version_mode = self._get_version_mode(mode=mode)
        return version_mode.pre_operations