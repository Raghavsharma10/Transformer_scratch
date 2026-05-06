def asVersion(self):
        """
        Convert the version data in this item to a
        L{twisted.python.versions.Version}.
        """
        return versions.Version(self.package, self.major, self.minor, self.micro)