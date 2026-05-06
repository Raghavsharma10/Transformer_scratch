def is_clean(self):
        """
        :data:`True` if the working tree (and index) is clean, :data:`False` otherwise.

        The implementation of :attr:`GitRepo.is_clean` checks whether ``git
        diff`` reports any differences. This command has several variants:

        1. ``git diff`` shows the difference between the index and working tree.
        2. ``git diff --cached`` shows the difference between the last commit and index.
        3. ``git diff HEAD`` shows the difference between the last commit and working tree.

        The implementation of :attr:`GitRepo.is_clean` uses the third command
        (``git diff HEAD``) in an attempt to hide the existence of git's index
        from callers that are trying to write code that works with Git and
        Mercurial using the same Python API.
        """
        # Make sure the local repository exists.
        self.create()
        # Check whether the `git diff HEAD' output is empty.
        listing = self.context.capture('git', 'diff', 'HEAD', check=False, silent=True)
        return len(listing.splitlines()) == 0