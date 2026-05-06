def find_revision_number(self, revision=None):
        """
        Find the local revision number of the given revision.

        .. note:: Bazaar has the concept of dotted revision numbers:

                   For revisions which have been merged into a branch, a dotted
                   notation is used (e.g., 3112.1.5). Dotted revision numbers
                   have three numbers. The first number indicates what mainline
                   revision change is derived from. The second number is the
                   branch counter. There can be many branches derived from the
                   same revision, so they all get a unique number. The third
                   number is the number of revisions since the branch started.
                   For example, 3112.1.5 is the first branch from revision
                   3112, the fifth revision on that branch.

                   (From http://doc.bazaar.canonical.com/bzr.2.6/en/user-guide/zen.html#understanding-revision-numbers)

                  However we really just want to give a bare integer to our
                  callers. It doesn't have to be globally accurate, but it
                  should increase as new commits are made. Below is the
                  equivalent of the git implementation for Bazaar.
        """
        # Make sure the local repository exists.
        self.create()
        # Try to find the revision number of the specified revision.
        revision = revision or self.default_revision
        output = self.context.capture('bzr', 'log', '--revision=..%s' % revision, '--line')
        revision_number = len([line for line in output.splitlines() if not is_empty_line(line)])
        if not (revision_number > 0):
            msg = "Failed to find local revision number! ('bzr log --line' gave unexpected output)"
            raise EnvironmentError(msg)
        return revision_number