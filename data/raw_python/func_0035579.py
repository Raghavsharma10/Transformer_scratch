def init_app(self, app):
        """Initialize Flask application."""
        if self.entry_point_group:
            eps = sorted(pkg_resources.iter_entry_points(
                self.entry_point_group), key=attrgetter('name'))
            for ep in eps:
                app.logger.debug("Loading config for entry point {}".format(
                    ep))
                app.config.from_object(ep.load())