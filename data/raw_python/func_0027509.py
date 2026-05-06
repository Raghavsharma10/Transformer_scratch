def is_in_team(self, team_id):
        """Test if user is in team"""

        if self.is_super_admin():
            return True
        team_id = uuid.UUID(str(team_id))
        return team_id in self.teams or team_id in self.child_teams_ids