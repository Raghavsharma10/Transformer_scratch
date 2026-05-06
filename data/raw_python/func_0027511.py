def is_feeder(self, team_id=None):
        """Ensure ther resource has the role FEEDER."""
        if team_id is None:
            return self._is_feeder
        team_id = uuid.UUID(str(team_id))
        if team_id not in self.teams_ids:
            return False
        return self.teams[team_id]['role'] == 'FEEDER'