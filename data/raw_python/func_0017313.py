def get_git_status(self):
        """
        Gets git and init versions and commits since the init version
        """
        ## get git branch
        self._get_git_branch()

        ## get tag in the init file
        self._get_init_release_tag()

        ## get log commits since <tag>
        try:
            self._get_log_commits()
        except Exception as inst:
            raise Exception(
        """
        Error: the version in __init__.py is {}, so 'git log' is 
        looking for commits that have happened since that version, but
        it appears there is not existing tag for that version. You may
        need to roll back the version in __init__.py to what is actually
        commited. Check with `git tag`.
        --------
        {}
        """.format(self.init_version, inst))

        ## where are we at?
        print("__init__.__version__ == '{}':".format(self.init_version))
        print("'{}' is {} commits ahead of origin/{}"
              .format(self.tag, len(self.commits), self.init_version))