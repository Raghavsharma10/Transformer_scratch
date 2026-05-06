def get_vcs_directory(context, directory):
        """Get the pathname of the directory containing the version control metadata files."""
        nested = os.path.join(directory, '.git')
        return nested if context.is_directory(nested) else directory