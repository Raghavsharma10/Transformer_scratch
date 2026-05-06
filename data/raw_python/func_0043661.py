def known_remotes(self):
        """The names of the configured remote repositories (a list of :class:`.Remote` objects)."""
        objects = []
        for line in self.context.capture('git', 'remote', '--verbose').splitlines():
            tokens = line.split()
            if len(tokens) >= 2:
                name = tokens[0]
                objects.append(Remote(
                    default=(name == 'origin'),
                    location=tokens[1], name=name, repository=self,
                    # We fall back to allowing both roles when we fail to
                    # recognize either role because:
                    #
                    #  1. This code is relatively new and may be buggy.
                    #  2. Practically speaking most git repositories will use
                    #     the same remote for pushing and pulling and in fact
                    #     this remote is likely to be the only remote :-).
                    roles=(['pull'] if '(fetch)' in tokens
                           else (['push'] if '(push)' in tokens
                           else (['push', 'pull']))),
                ))
        return objects