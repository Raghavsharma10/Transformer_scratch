def on_created(self, event):
        """
        Called when a new file or directory is created.

        Todo:
            This should be also used (extended from another class?) to watch
            for some special name file (like ".boussole-watcher-stop" create to
            raise a KeyboardInterrupt, so we may be able to unittest the
            watcher (click.CliRunner is not able to send signal like CTRL+C
            that is required to watchdog observer loop)

        Args:
            event: Watchdog event, either ``watchdog.events.DirCreatedEvent``
                or ``watchdog.events.FileCreatedEvent``.
        """
        if not self._event_error:
            self.logger.info(u"Change detected from a create on: %s",
                             event.src_path)

            self.compile_dependencies(event.src_path)