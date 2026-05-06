def _get_version_mode(self, mode=None):
        """Return a VersionMode for a mode name.

        When the mode is None, we are working with the 'base' mode.
        """
        version_mode = self._version_modes.get(mode)
        if not version_mode:
            version_mode = self._version_modes[mode] = VersionMode(name=mode)
        return version_mode