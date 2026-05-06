def contains_repository(cls, context, directory):
        """Check whether the given directory contains a local repository."""
        directory = cls.get_vcs_directory(context, directory)
        return context.is_file(os.path.join(directory, 'config'))