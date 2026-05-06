def contribute_error_pages(self):
        """Contributes generic static error massage pages to an existing section."""

        static_dir = self.settings.STATIC_ROOT

        if not static_dir:
            # Source static directory is not configured. Use temporary.
            import tempfile
            static_dir = os.path.join(tempfile.gettempdir(), self.project_name)
            self.settings.STATIC_ROOT = static_dir

        self.section.routing.set_error_pages(
            common_prefix=os.path.join(static_dir, 'uwsgify'))