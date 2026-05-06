def edit(self, title=None, body=None, assignee=None, state=None,
             milestone=None, labels=None):
        """Edit this issue.

        :param str title: Title of the issue
        :param str body: markdown formatted body (description) of the issue
        :param str assignee: login name of user the issue should be assigned
            to
        :param str state: accepted values: ('open', 'closed')
        :param int milestone: the NUMBER (not title) of the milestone to
            assign this to [1]_, or 0 to remove the milestone
        :param list labels: list of labels to apply this to
        :returns: bool

        .. [1] Milestone numbering starts at 1, i.e. the first milestone you
               create is 1, the second is 2, etc.
        """
        json = None
        data = {'title': title, 'body': body, 'assignee': assignee,
                'state': state, 'milestone': milestone, 'labels': labels}
        self._remove_none(data)
        if data:
            if 'milestone' in data and data['milestone'] == 0:
                data['milestone'] = None
            json = self._json(self._patch(self._api, data=dumps(data)), 200)
        if json:
            self._update_(json)
            return True
        return False