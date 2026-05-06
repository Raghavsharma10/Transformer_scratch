def depends_on(self, dependency):
        """
        List of packages that depend on dependency
        :param dependency: package name, e.g.  'vext' or 'Pillow'
        """
        packages = self.package_info()
        return [package for package in packages if dependency in package.get("requires", "")]