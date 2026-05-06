def get_scss_files(self, skip_partials=True, with_source_path=False):
        """Gets all SCSS files in the source directory.

        :param bool skip_partials: If True, partials will be ignored. Otherwise,
                                   all SCSS files, including ones that begin
                                   with '_' will be returned.
        :param boom with_source_path: If true, the `source_path` will be added
                                      to all of the paths. Otherwise, it will
                                      be stripped.
        :returns: A list of the SCSS files in the source directory

        """
        scss_files = []

        for root, dirs, files in os.walk(self._source_path):
            for filename in fnmatch.filter(files, "*.scss"):
                if filename.startswith("_") and skip_partials:
                    continue

                full_path = os.path.join(root, filename)
                if not with_source_path:
                    full_path = full_path.split(self._source_path)[1]

                    if full_path.startswith("/"):
                        full_path = full_path[1:]

                scss_files.append(full_path)

        return scss_files