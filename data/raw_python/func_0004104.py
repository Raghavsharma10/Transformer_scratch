def _watch_refresh_source(self, event):
        """Refresh sources then templates
        """
        self.logger.info("Sources changed...")

        try:
            self.sources = self._get_sources()
            self._render_template(self.sources)
        except:
            pass