def work(self):
        """
        A list of :class:`Employment` instances describing the user's work history.

        Each structure has attributes ``employer``, ``position``, ``started_at`` and ``ended_at``.

        ``employer`` and ``position`` reference ``Page`` instances, while ``started_at`` and ``ended_at``
        are datetime objects.
        """
        employments = []

        for work in self.cache['work']:
            employment = Employment(
                employer = work.get('employer'),
                position = work.get('position'),
                started_at = work.get('start_date'),
                ended_at = work.get('end_date')
            )

            employments.append(employment)

        return employments