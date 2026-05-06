def is_noop(self):
        """Check if version is a no operation version.
        """
        has_operations = [mode.pre_operations or mode.post_operations
                          for mode in self._version_modes.values()]
        has_upgrade_addons = [mode.upgrade_addons or mode.remove_addons
                              for mode in self._version_modes.values()]
        noop = not any((has_upgrade_addons, has_operations))
        return noop