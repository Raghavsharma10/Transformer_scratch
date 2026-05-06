def _watch_refresh_template(self, event):
        """Refresh template's contents
        """
        self.logger.info("Template changed...")

        try:
            self._render_template(self.sources)
        except:
            pass