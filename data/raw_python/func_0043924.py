def tags(self):
        """
        A dictionary that maps tag names to :class:`Revision` objects.

        Here's an example based on a mirror of the git project's repository:

        >>> from pprint import pprint
        >>> from vcs_repo_mgr.backends.git import GitRepo
        >>> repository = GitRepo(remote='https://github.com/git/git.git')
        >>> pprint(repository.tags)
        {'v0.99': Revision(repository=GitRepo(...),
                           tag='v0.99',
                           revision_id='d6602ec5194c87b0fc87103ca4d67251c76f233a'),
         'v0.99.1': Revision(repository=GitRepo(...),
                             tag='v0.99.1',
                             revision_id='f25a265a342aed6041ab0cc484224d9ca54b6f41'),
         'v0.99.2': Revision(repository=GitRepo(...),
                             tag='v0.99.2',
                             revision_id='c5db5456ae3b0873fc659c19fafdde22313cc441'),
         ..., # dozens of tags omitted to keep this example short
         'v2.3.6': Revision(repository=GitRepo(...),
                            tag='v2.3.6',
                            revision_id='8e7304597727126cdc52771a9091d7075a70cc31'),
         'v2.3.7': Revision(repository=GitRepo(...),
                            tag='v2.3.7',
                            revision_id='b17db4d9c966de30f5445632411c932150e2ad2f'),
         'v2.4.0': Revision(repository=GitRepo(...),
                            tag='v2.4.0',
                            revision_id='67308bd628c6235dbc1bad60c9ad1f2d27d576cc')}
        """
        # Make sure the local repository exists.
        self.create()
        # Create a mapping of tag names to revisions.
        return dict((r.tag, r) for r in self.find_tags())