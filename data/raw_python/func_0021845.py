def versions_available(self):
        """
        Query PyPI for a particular version or all versions of a package

        @returns: 0 if version(s) found or 1 if none found
        """

        if self.version:
            spec = "%s==%s" % (self.project_name, self.version)
        else:
            spec = self.project_name

        if self.all_versions and self.version in self.all_versions:
            print_pkg_versions(self.project_name, [self.version])
        elif not self.version and self.all_versions:
            print_pkg_versions(self.project_name, self.all_versions)
        else:
            if self.version:
                self.logger.error("No pacakge found for version %s" \
                        % self.version)
            else:
                self.logger.error("No pacakge found for %s" % self.project_name)
            return 1
        return 0