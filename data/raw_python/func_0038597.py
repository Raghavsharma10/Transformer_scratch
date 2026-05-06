def send_contributor_email(self, contributor):
        """Send an EmailMessage object for a given contributor."""
        ContributorReport(
            contributor,
            month=self.month,
            year=self.year,
            deadline=self._deadline,
            start=self._start,
            end=self._end
        ).send()