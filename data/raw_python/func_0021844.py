def query_metadata_pypi(self):
        """
        Show pkg metadata queried from PyPI

        @returns: 0

        """
        if self.version and self.version in self.all_versions:
            metadata = self.pypi.release_data(self.project_name, self.version)
        else:
            #Give highest version
            metadata = self.pypi.release_data(self.project_name, \
                    self.all_versions[0])

        if metadata:
            for key in metadata.keys():
                if not self.options.fields or (self.options.fields and \
                        self.options.fields==key):
                    print("%s: %s" % (key, metadata[key]))
        return 0