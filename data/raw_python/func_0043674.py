def get_merge_command(self, revision):
        """Get the command to merge a revision into the current branch (without committing the result)."""
        return [
            'git',
            '-c', 'user.name=%s' % self.author.name,
            '-c', 'user.email=%s' % self.author.email,
            'merge', '--no-commit', '--no-ff',
            revision,
        ]