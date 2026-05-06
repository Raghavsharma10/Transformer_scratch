def _get_allowed_sections(self, dashboard):
        """
        Get the sections to display based on dashboard
        """

        allowed_titles = [x[0] for x in dashboard]
        allowed_sections = [x[2] for x in dashboard]
        return tuple(allowed_sections), tuple(allowed_titles)