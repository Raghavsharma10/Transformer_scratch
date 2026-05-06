def is_valid(self):
        """returns `True` if the report should be sent."""
        if not self.total:
            return False
        if not self.contributor.freelanceprofile.is_freelance:
            return False
        return True