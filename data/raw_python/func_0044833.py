def resolve(self, sourcepath, paths, library_paths=None):
        """
        Resolve given paths from given base paths

        Return resolved path list.

        Note:
            Resolving strategy is made like libsass do, meaning paths in
            import rules are resolved from the source file where the import
            rules have been finded.

            If import rule is not explicit enough and two file are candidates
            for the same rule, it will raises an error. But contrary to
            libsass, this happen also for files from given libraries in
            ``library_paths`` (oposed to libsass just silently taking the
            first candidate).

        Args:
            sourcepath (str): Source file path, its directory is used to
                resolve given paths. The path must be an absolute path to
                avoid errors on resolving.
            paths (list): Relative paths (from ``sourcepath``) to resolve.
            library_paths (list): List of directory paths for libraries to
                resolve paths if resolving fails on the base source path.
                Default to None.

        Raises:
            UnresolvablePath: If a path does not exist and
                ``STRICT_PATH_VALIDATION`` attribute is ``True``.

        Returns:
            list: List of resolved path.
        """
        # Split basedir/filename from sourcepath, so the first resolving
        # basepath is the sourcepath directory, then the optionnal
        # given libraries
        basedir, filename = os.path.split(sourcepath)
        basepaths = [basedir]
        resolved_paths = []

        # Add given library paths to the basepaths for resolving
        # Accept a string if not allready in basepaths
        if library_paths and isinstance(library_paths, string_types) and \
           library_paths not in basepaths:
            basepaths.append(library_paths)
        # Add path item from list if not allready in basepaths
        elif library_paths:
            for k in list(library_paths):
                if k not in basepaths:
                    basepaths.append(k)

        for import_rule in paths:
            candidates = self.candidate_paths(import_rule)

            # Search all existing candidates:
            # * If more than one candidate raise an error;
            # * If only one, accept it;
            # * If no existing candidate raise an error;
            stack = []
            for i, basepath in enumerate(basepaths):
                checked = self.check_candidate_exists(basepath, candidates)
                if checked:
                    stack.extend(checked)

            # More than one existing candidate
            if len(stack) > 1:
                raise UnclearResolution(
                    "rule '{}' This is not clear for these paths: {}".format(
                        import_rule, ', '.join(stack)
                    )
                )
            # Accept the single one
            elif len(stack) == 1:
                resolved_paths.append(os.path.normpath(stack[0]))
            # No validated candidate
            else:
                if self.STRICT_PATH_VALIDATION:
                    raise UnresolvablePath(
                        "Imported path '{}' does not exist in '{}'".format(
                            import_rule, basedir
                        )
                    )

        return resolved_paths