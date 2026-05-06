def hairball_files(self, paths, extensions):
        """Yield filepath to files with the proper extension within paths."""
        def add_file(filename):
            return os.path.splitext(filename)[1] in extensions

        while paths:
            arg_path = paths.pop(0)
            if os.path.isdir(arg_path):
                found = False
                for path, dirs, files in os.walk(arg_path):
                    dirs.sort()  # Traverse in sorted order
                    for filename in sorted(files):
                        if add_file(filename):
                            yield os.path.join(path, filename)
                            found = True
                if not found:
                    if not self.options.quiet:
                        print('No files found in {}'.format(arg_path))
            elif add_file(arg_path):
                yield arg_path
            elif not self.options.quiet:
                print('Invalid file {}'.format(arg_path))
                print('Did you forget to load a Kurt plugin (-k)?')