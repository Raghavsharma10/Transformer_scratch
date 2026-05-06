def _patch_expand_path(self, settings, name, value):
        """
        Patch a path to expand home directory and make absolute path.

        Args:
            settings (dict): Current settings.
            name (str): Setting name.
            value (str): Path to patch.

        Returns:
            str: Patched path to an absolute path.

        """
        if os.path.isabs(value):
            return os.path.normpath(value)

        # Expand home directory if any
        value = os.path.expanduser(value)

        # If the path is not yet an absolute directory, make it so from base
        # directory if not empty
        if not os.path.isabs(value) and self.projectdir:
            value = os.path.join(self.projectdir, value)

        return os.path.normpath(value)