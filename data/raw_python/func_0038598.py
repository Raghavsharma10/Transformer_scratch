def send_mass_contributor_emails(self):
        """Send report email to all relevant contributors."""
        # If the report configuration is not active we only send to the debugging user.
        for contributor in self.contributors:
            if contributor.email not in EMAIL_SETTINGS.get("EXCLUDED", []):
                self.send_contributor_email(contributor)