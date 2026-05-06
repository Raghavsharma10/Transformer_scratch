def is_product_owner(self, team_id):
        """Ensure the user is a PRODUCT_OWNER."""

        if self.is_super_admin():
            return True
        team_id = uuid.UUID(str(team_id))
        return team_id in self.child_teams_ids