def from_str(cls, version_str: str):
        """
        Alternate constructor that accepts a string SemVer.
        """
        o = cls()
        o.version = version_str
        return o