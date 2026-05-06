def _add_static_handlers(self, handlers):
        """
        Creates and adds the handles needed for serving static files.

        :param handlers:
        """
        for url, path in self.static_dirs:
            handlers.append((url.rstrip("/") + "/(.*)", StaticFileHandler, {"path": path}))