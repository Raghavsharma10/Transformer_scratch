def on_modified(self, event):
        """
        Called when a file or directory is modified.

        Args:
            event: Watchdog event, ``watchdog.events.DirModifiedEvent`` or
                ``watchdog.events.FileModifiedEvent``.
        """
        if not self._event_error:
            self.logger.info(u"Change detected from an edit on: %s",
                             event.src_path)

            self.compile_dependencies(event.src_path)