def create_issue(self,
                     title,
                     body=None,
                     assignee=None,
                     milestone=None,
                     labels=None):
        """Creates an issue on this repository.

        :param str title: (required), title of the issue
        :param str body: (optional), body of the issue
        :param str assignee: (optional), login of the user to assign the
            issue to
        :param int milestone: (optional), id number of the milestone to
            attribute this issue to (e.g. ``m`` is a :class:`Milestone
            <github3.issues.milestone.Milestone>` object, ``m.number`` is
            what you pass here.)
        :param labels: (optional), labels to apply to this
            issue
        :type labels: list of strings
        :returns: :class:`Issue <github3.issues.issue.Issue>` if successful,
            otherwise None
        """
        issue = {'title': title, 'body': body, 'assignee': assignee,
                 'milestone': milestone, 'labels': labels}
        self._remove_none(issue)
        json = None

        if issue:
            url = self._build_url('issues', base_url=self._api)
            json = self._json(self._post(url, data=issue), 201)

        return Issue(json, self) if json else None