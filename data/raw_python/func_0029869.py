def _init_view(self):
        """
        Initialize View with project settings.
        """
        views_engine = get_config('rails.views.engine', 'jinja')
        templates_dir = os.path.join(self._project_dir, "views", "templates")
        self._view = View(views_engine, templates_dir)