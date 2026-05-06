def pull_request(self, owner, repository, number):
        """Fetch pull_request #:number: from :owner:/:repository

        :param str owner: (required), owner of the repository
        :param str repository: (required), name of the repository
        :param int number: (required), issue number
        :return: :class:`Issue <github3.issues.Issue>`
        """
        r = self.repository(owner, repository)
        return r.pull_request(number) if r else None