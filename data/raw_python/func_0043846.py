def get_commit_command(self, message, author=None):
        """
        Get the command to commit changes to tracked files in the working tree.

        This method uses the ``hg remove --after`` to match the semantics of
        ``git commit --all`` (which is _not_ the same as ``hg commit
        --addremove``) however ``hg remove --after`` is _very_ verbose (it
        comments on every existing file in the repository) and it ignores the
        ``--quiet`` option. This explains why I've decided to silence the
        standard error stream (though I feel I may regret this later).
        """
        tokens = ['hg remove --after 2>/dev/null; hg commit']
        if author:
            tokens.append('--user=%s' % quote(author.combined))
        tokens.append('--message=%s' % quote(message))
        return [' '.join(tokens)]