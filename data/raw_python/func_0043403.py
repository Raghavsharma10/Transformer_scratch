def _get_patch_path(cls, patch_name):
        """Returns the absolute path for patch *patch_name*."""
        return os.path.join(cls.STASH_PATH, patch_name) if patch_name else None