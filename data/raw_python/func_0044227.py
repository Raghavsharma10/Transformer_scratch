def on_moved(self, event):
        """
        Called when a file or a directory is moved or renamed.

        Many editors don't directly change a file, instead they make a
        transitional file like ``*.part`` then move it to the final filename.

        Args:
            event: Watchdog event, either ``watchdog.events.DirMovedEvent`` or
                ``watchdog.events.FileModifiedEvent``.
        """
        if not self._event_error:
            # We are only interested for final file, not transitional file
            # from editors (like *.part)
            pathtools_options = {
                'included_patterns': self.patterns,
                'excluded_patterns': self.ignore_patterns,
                'case_sensitive': self.case_sensitive,
            }
            # Apply pathtool matching on destination since Watchdog only
            # automatically apply it on source
            if match_path(event.dest_path, **pathtools_options):
                self.logger.info(u"Change detected from a move on: %s",
                                 event.dest_path)
                self.compile_dependencies(event.dest_path)