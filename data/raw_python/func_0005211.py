def get(cls, sha1=''):
        # type: (str) -> CommitDetails
        """ Return details about a given commit.

        Args:
            sha1 (str):
                The sha1 of the commit to query. If not given, it will return
                the details for the latest commit.

        Returns:
            CommitDetails: Commit details. You can use the instance of the
            class to query git tree further.
        """
        with conf.within_proj_dir():
            cmd = 'git show -s --format="%H||%an||%ae||%s||%b||%P" {}'.format(
                sha1
            )
            result = shell.run(cmd, capture=True, never_pretend=True).stdout

        sha1, name, email, title, desc, parents = result.split('||')

        return CommitDetails(
            sha1=sha1,
            author=Author(name, email),
            title=title,
            desc=desc,
            parents_sha1=parents.split(),
        )