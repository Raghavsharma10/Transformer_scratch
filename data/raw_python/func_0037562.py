def available_updates(self, obj):
        """Print out all versions ahead of the current one."""
        from package_monitor import pypi
        package = pypi.Package(obj.package_name)
        versions = package.all_versions()
        return html_list([v for v in versions if v > obj.current_version])