def compilable_sources(self, sourcedir, absolute=False, recursive=True,
                           excludes=[]):
        """
        Find all scss sources that should be compiled, aka all sources that
        are not "partials" Sass sources.


        Args:
            sourcedir (str): Directory path to scan.

        Keyword Arguments:
            absolute (bool): Returned paths will be absolute using
                ``sourcedir`` argument (if True), else return relative paths.
            recursive (bool): Switch to enabled recursive finding (if True).
                Default to True.
            excludes (list): A list of excluding patterns (glob patterns).
                Patterns are matched against the relative filepath (from its
                sourcedir).

        Returns:
            list: List of source paths.
        """
        filepaths = []

        for root, dirs, files in os.walk(sourcedir):
            # Sort structure to avoid arbitrary order
            dirs.sort()
            files.sort()
            for item in files:
                # Store relative directory but drop it if at root ('.')
                relative_dir = os.path.relpath(root, sourcedir)
                if relative_dir == '.':
                    relative_dir = ''

                # Matching all conditions
                absolute_filepath = os.path.join(root, item)
                conditions = {
                    'sourcedir': sourcedir,
                    'nopartial': True,
                    'exclude_patterns': excludes,
                    'excluded_libdirs': [],
                }
                if self.match_conditions(absolute_filepath, **conditions):
                    relative_filepath = os.path.join(relative_dir, item)

                    if absolute:
                        filepath = absolute_filepath
                    else:
                        filepath = relative_filepath

                    filepaths.append(filepath)

            # For non recursive usage, break from the first entry
            if not recursive:
                break

        return filepaths