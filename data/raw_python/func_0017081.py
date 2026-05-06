def create_milestone(self, title, state=None, description=None,
                         due_on=None):
        """Create a milestone for this repository.

        :param str title: (required), title of the milestone
        :param str state: (optional), state of the milestone, accepted
            values: ('open', 'closed'), default: 'open'
        :param str description: (optional), description of the milestone
        :param str due_on: (optional), ISO 8601 formatted due date
        :returns: :class:`Milestone <github3.issues.milestone.Milestone>` if
            successful, otherwise None
        """
        url = self._build_url('milestones', base_url=self._api)
        if state not in ('open', 'closed'):
            state = None
        data = {'title': title, 'state': state,
                'description': description, 'due_on': due_on}
        self._remove_none(data)
        json = None
        if data:
            json = self._json(self._post(url, data=data), 201)
        return Milestone(json, self) if json else None