def releases(self):
        r"""
        A dictionary that maps release identifiers to :class:`Release` objects.

        Here's an example based on a mirror of the git project's repository
        which shows the last ten releases based on tags, where each release
        identifier captures a tag without its 'v' prefix:

        >>> from pprint import pprint
        >>> from vcs_repo_mgr.backends.git import GitRepo
        >>> repository = GitRepo(remote='https://github.com/git/git.git',
        ...                      release_scheme='tags',
        ...                      release_filter=r'^v(\d+(?:\.\d+)*)$')
        >>> pprint(repository.ordered_releases[-10:])
        [Release(revision=Revision(..., tag='v2.2.2', ...), identifier='2.2.2'),
         Release(revision=Revision(..., tag='v2.3.0', ...), identifier='2.3.0'),
         Release(revision=Revision(..., tag='v2.3.1', ...), identifier='2.3.1'),
         Release(revision=Revision(..., tag='v2.3.2', ...), identifier='2.3.2'),
         Release(revision=Revision(..., tag='v2.3.3', ...), identifier='2.3.3'),
         Release(revision=Revision(..., tag='v2.3.4', ...), identifier='2.3.4'),
         Release(revision=Revision(..., tag='v2.3.5', ...), identifier='2.3.5'),
         Release(revision=Revision(..., tag='v2.3.6', ...), identifier='2.3.6'),
         Release(revision=Revision(..., tag='v2.3.7', ...), identifier='2.3.7'),
         Release(revision=Revision(..., tag='v2.4.0', ...), identifier='2.4.0')]
        """
        available_releases = {}
        available_revisions = getattr(self, self.release_scheme)
        for identifier, revision in available_revisions.items():
            match = self.compiled_filter.match(identifier)
            if match:
                # If the regular expression contains a capturing group we
                # set the release identifier to the captured substring
                # instead of the complete tag/branch identifier.
                captures = match.groups()
                if captures:
                    identifier = captures[0]
                available_releases[identifier] = Release(
                    revision=revision,
                    identifier=identifier,
                )
        return available_releases