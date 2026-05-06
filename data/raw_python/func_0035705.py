def get_members(cls, session, team_or_id):
        """List the members for the team.

        Args:
            team_or_id (helpscout.models.Person or int): Team or the ID of
                the team to get the folders for.

        Returns:
            RequestPaginator(output_type=helpscout.models.Users): Users
                iterator.
        """
        if isinstance(team_or_id, Person):
            team_or_id = team_or_id.id
        return cls(
            '/teams/%d/members.json' % team_or_id,
            session=session,
            out_type=User,
        )