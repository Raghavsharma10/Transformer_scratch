def case_sensitive_name(self, package_name):
        """
        Return case-sensitive package name given any-case package name

        @param project_name: PyPI project name
        @type project_name: string

        """
        if len(self.environment[package_name]):
            return self.environment[package_name][0].project_name