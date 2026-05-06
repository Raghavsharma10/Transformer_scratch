def output_files(self):
        """Returns the list of output files from this rule.

        Paths are relative to buildroot.
        """
        for item in self.source_files:
            yield os.path.join(self.address.repo, self.address.path, item)