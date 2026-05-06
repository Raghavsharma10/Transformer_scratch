def known_remotes(self):
        """The names of the configured remote repositories (a list of :class:`.Remote` objects)."""
        objects = []
        for line in self.context.capture('hg', 'paths').splitlines():
            name, _, location = line.partition('=')
            if name and location:
                name = name.strip()
                objects.append(Remote(
                    default=(name in ('default', 'default-push')),
                    location=location.strip(), name=name, repository=self,
                    # We give the `default-push' remote the `push' role only,
                    # while allowing both roles for other remotes. This isn't
                    # strictly speaking correct but it will prevent
                    # Repository.pull() from considering the `default-push'
                    # remote as a suitable default to pull from (which is not
                    # what Mercurial does when you run `hg pull').
                    roles=(['push'] if name == 'default-push' else ['push', 'pull']),
                ))
        return objects