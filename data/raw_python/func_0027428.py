def setup(self):
        """Initialize filter just before it will be used."""
        super(CleanCSSFilter, self).setup()
        self.root = current_app.config.get('COLLECT_STATIC_ROOT')