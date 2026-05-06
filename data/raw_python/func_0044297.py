def inspect(self, *args, **kwargs):
        """
        Recursively inspect all given SCSS files to find imported dependencies.

        This does not return anything. Just fill internal buffers about
        inspected files.

        Note:
            This will ignore orphan files (files that are not imported from
            any of given SCSS files).

        Args:
            *args: One or multiple arguments, each one for a source file path
                to inspect.

        Keyword Arguments:
            library_paths (list): List of directory paths for libraries to
                resolve paths if resolving fails on the base source path.
                Default to None.
        """
        library_paths = kwargs.get('library_paths', None)

        for sourcepath in args:
            self.look_source(sourcepath, library_paths=library_paths)