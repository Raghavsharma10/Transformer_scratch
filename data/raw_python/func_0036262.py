def get(self, bug_number):
        """
            Get a bug from Bugzilla. If there is a login token created during
            object initialisation it will be part of the query string passed to
            Bugzilla

            :param bug_number: Bug Number that will be searched. If found will
                               return a Bug object.

            >>> bugzilla = Bugsy()
            >>> bug = bugzilla.get(123456)
        """
        bug = self.request(
            'bug/%s' % bug_number,
            params={"include_fields": self. DEFAULT_SEARCH}
        )
        return Bug(self, **bug['bugs'][0])