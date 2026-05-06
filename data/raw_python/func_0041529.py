def find_files(self, root):
        """
        Helper method to get all files in the given root.
        """

        def is_ignored(path, ignore_patterns):
            """
            Check if the given path should be ignored or not.
            """
            filename = os.path.basename(path)
            ignore = lambda pattern: fnmatch.fnmatchcase(filename, pattern)
            return any(ignore(pattern) for pattern in ignore_patterns)

        dir_suffix = '%s*' % os.sep
        normalized_patterns = [
            p[:-len(dir_suffix)] if p.endswith(dir_suffix) else p
            for p in self.ignore_patterns
        ]

        all_files = []
        walker = os.walk(root, topdown=True, followlinks=self.follow_symlinks)
        for dir_path, dir_names, file_names in walker:
            for dir_name in dir_names[:]:
                path = os.path.normpath(os.path.join(dir_path, dir_name))
                if is_ignored(path, normalized_patterns):
                    dir_names.remove(dir_name)
                    if self.verbose:
                        print_out("Ignoring directory '{:}'".format(dir_name))
            for file_name in file_names:
                path = os.path.normpath(os.path.join(dir_path, file_name))
                if is_ignored(path, self.ignore_patterns):
                    if self.verbose:
                        print_out("Ignoring file '{:}' in '{:}'".format(
                                  file_name, dir_path))
                else:
                    all_files.append((dir_path, file_name))
        return sorted(all_files)