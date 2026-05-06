def branches(self):
        # type: () -> List[str]
        """ List of all branches this commit is a part of. """
        if self._branches is None:
            cmd = 'git branch --contains {}'.format(self.sha1)
            out = shell.run(
                cmd,
                capture=True,
                never_pretend=True
            ).stdout.strip()
            self._branches = [x.strip('* \t\n') for x in out.splitlines()]

        return self._branches