def custom_template_name(self):
        """
        Returns the path for the custom special coverage template we want.
        """
        base_path = getattr(settings, "CUSTOM_SPECIAL_COVERAGE_PATH", "special_coverage/custom")
        if base_path is None:
            base_path = ""
        return "{0}/{1}_custom.html".format(
            base_path, self.slug.replace("-", "_")
        ).lstrip("/")