def branches(self):
        """
        A dictionary that maps branch names to :class:`Revision` objects.

        Here's an example based on a mirror of the git project's repository:

        >>> from pprint import pprint
        >>> from vcs_repo_mgr.backends.git import GitRepo
        >>> repository = GitRepo(remote='https://github.com/git/git.git')
        >>> pprint(repository.branches)
        {'maint':  Revision(repository=GitRepo(...), branch='maint',  revision_id='16018ae'),
         'master': Revision(repository=GitRepo(...), branch='master', revision_id='8440f74'),
         'next':   Revision(repository=GitRepo(...), branch='next',   revision_id='38e7071'),
         'pu':     Revision(repository=GitRepo(...), branch='pu',     revision_id='d61c1fa'),
         'todo':   Revision(repository=GitRepo(...), branch='todo',   revision_id='dea8a2d')}
        """
        # Make sure the local repository exists.
        self.create()
        # Create a mapping of branch names to revisions.
        return dict((r.branch, r) for r in self.find_branches())