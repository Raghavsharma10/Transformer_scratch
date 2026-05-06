def get_delete_branch_command(self, branch_name, message, author):
        """Get the command to delete or close a branch in the local repository."""
        tokens = ['hg update --rev=%s && hg commit' % quote(branch_name)]
        if author:
            tokens.append('--user=%s' % quote(author.combined))
        tokens.append('--message=%s' % quote(message))
        tokens.append('--close-branch')
        return [' '.join(tokens)]