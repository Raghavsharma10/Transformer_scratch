def match(self, location):
        """
        Check if the given location "matches".

        :param location: The :class:`Location` object to try to match.
        :returns: :data:`True` if the two locations are on the same system and
                  the :attr:`directory` can be matched as a filename pattern or
                  a literal match on the normalized pathname.
        """
        if self.ssh_alias != location.ssh_alias:
            # Never match locations on other systems.
            return False
        elif self.have_wildcards:
            # Match filename patterns using fnmatch().
            return fnmatch.fnmatch(location.directory, self.directory)
        else:
            # Compare normalized directory pathnames.
            self = os.path.normpath(self.directory)
            other = os.path.normpath(location.directory)
            return self == other