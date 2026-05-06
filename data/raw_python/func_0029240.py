def key_to_path(self, key):
        """Return the fullpath to the file with sha1sum key."""
        return os.path.join(self.cache_dir, key[:2], key[2:4],
                            key[4:] + '.pkl')