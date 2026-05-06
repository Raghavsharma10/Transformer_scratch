def issue(self, owner, repository, number):
        """Fetch issue #:number: from https://github.com/:owner:/:repository:

        :param str owner: (required), owner of the repository
        :param str repository: (required), name of the repository
        :param int number: (required), issue number
        :return: :class:`Issue <github3.issues.Issue>`
        """
        repo = self.repository(owner, repository)
        if repo:
            return repo.issue(number)
        return None