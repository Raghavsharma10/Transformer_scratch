def set_timezone(self, request, org):
        """Set the current timezone from the org configuration."""
        if org and org.timezone:
            timezone.activate(org.timezone)