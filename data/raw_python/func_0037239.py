def update_from_pypi(self):
        """Call get_latest_version and then save the object."""
        package = pypi.Package(self.package_name)
        self.licence = package.licence()
        if self.is_parseable:
            self.latest_version = package.latest_version()
            self.next_version = package.next_version(self.current_version)
            self.diff_status = pypi.version_diff(self.current_version, self.latest_version)
            self.python_support = package.python_support()
            self.django_support = package.django_support()
            self.supports_py3 = package.supports_py3()
        self.checked_pypi_at = tz_now()
        self.save()
        return self