def get_root_path(self, path):
        """See :py:meth:`~stash.repository.Repository.get_root_path`."""
        # Look at the directories present in the current working directory. In
        # case a .svn directory is present, we know we are in the root directory
        # of a Subversion repository (for Subversion 1.7.x). In case no
        # repository specific folder is found, and the current directory has a
        # parent directory, look if a repository specific directory can be found
        # in the parent directory.
        while path != '/':
            if '.svn' in os.listdir(path):
                return path
            path = os.path.abspath(os.path.join(path, os.pardir))

        # No Subversion repository found.
        return None