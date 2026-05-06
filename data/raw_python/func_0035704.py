def get(cls, session, team_id):
        """Return a specific team.

        Args:
            session (requests.sessions.Session): Authenticated session.
            team_id (int): The ID of the team to get.

        Returns:
            helpscout.models.Person: A person singleton representing the team,
                if existing. Otherwise ``None``.
        """
        return cls(
            '/teams/%d.json' % team_id,
            singleton=True,
            session=session,
        )