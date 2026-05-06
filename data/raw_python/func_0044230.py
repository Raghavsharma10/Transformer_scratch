def on_deleted(self, event):
        """
        Called when a file or directory is deleted.

        Todo:
            May be bugged with inspector and sass compiler since the does not
            exists anymore.

        Args:
            event: Watchdog event, ``watchdog.events.DirDeletedEvent`` or
                ``watchdog.events.FileDeletedEvent``.
        """
        if not self._event_error:
            self.logger.info(u"Change detected from deletion of: %s",
                             event.src_path)
            # Never try to compile the deleted source
            self.compile_dependencies(event.src_path, include_self=False)