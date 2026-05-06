def create_issue(self,
                     owner,
                     repository,
                     title,
                     body=None,
                     assignee=None,
                     milestone=None,
                     labels=[]):
        """Create an issue on the project 'repository' owned by 'owner'
        with title 'title'.

        body, assignee, milestone, labels are all optional.

        :param str owner: (required), login of the owner
        :param str repository: (required), repository name
        :param str title: (required), Title of issue to be created
        :param str body: (optional), The text of the issue, markdown
            formatted
        :param str assignee: (optional), Login of person to assign
            the issue to
        :param int milestone: (optional), id number of the milestone to
            attribute this issue to (e.g. ``m`` is a :class:`Milestone
            <github3.issues.Milestone>` object, ``m.number`` is what you pass
            here.)
        :param list labels: (optional), List of label names.
        :returns: :class:`Issue <github3.issues.Issue>` if successful, else
            None
        """
        repo = None
        if owner and repository and title:
            repo = self.repository(owner, repository)

        if repo:
            return repo.create_issue(title, body, assignee, milestone, labels)

        # Regardless, something went wrong. We were unable to create the
        # issue
        return None