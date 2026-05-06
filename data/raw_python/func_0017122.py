def milestone(self, number):
        """Get the milestone indicated by ``number``.

        :param int number: (required), unique id number of the milestone
        :returns: :class:`Milestone <github3.issues.milestone.Milestone>`
        """
        json = None
        if int(number) > 0:
            url = self._build_url('milestones', str(number),
                                  base_url=self._api)
            json = self._json(self._get(url), 200)
        return Milestone(json, self) if json else None