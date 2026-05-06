def _gen_full_path(self, filename, file_system=None):
        """Generate full file path on remote device."""
        if file_system is None:
            return '{}/{}'.format(self.dest_file_system, filename)
        else:
            if ":" not in file_system:
                raise ValueError("Invalid file_system specified: {}".format(file_system))
            return '{}/{}'.format(file_system, filename)