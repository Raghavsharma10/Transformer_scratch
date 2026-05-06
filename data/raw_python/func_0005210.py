def number(self):
        # type: () -> int
        """ Return this commits number.

        This is the same as the total number of commits in history up until
        this commit.

        This value can be useful in some CI scenarios as it allows to track
        progress on any given branch (although there can be two commits with the
        same number existing on different branches).

        Returns:
            int: The commit number/index.
        """
        cmd = 'git log --oneline {}'.format(self.sha1)
        out = shell.run(cmd, capture=True, never_pretend=True).stdout.strip()
        return len(out.splitlines())