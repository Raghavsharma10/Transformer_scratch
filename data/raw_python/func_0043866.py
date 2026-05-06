def known_remotes(self):
        """The names of the configured remote repositories (a list of :class:`.Remote` objects)."""
        objects = []
        output = self.context.capture(
            'bzr', 'config', 'parent_location',
            check=False, silent=True,
        )
        if output and not output.isspace():
            location = output.strip()
            # The `bzr branch' command has the unusual habit of converting
            # absolute pathnames into relative pathnames. Although I get why
            # this can be preferred over the use of absolute pathnames I
            # nevertheless want vcs-repo-mgr to communicate to its callers as
            # unambiguously as possible, so if we detect a relative pathname
            # we convert it to an absolute pathname.
            if location.startswith('../'):
                location = os.path.normpath(os.path.join(self.local, location))
            objects.append(Remote(
                default=True,
                location=location,
                repository=self,
                roles=['push', 'pull'],
            ))
        return objects