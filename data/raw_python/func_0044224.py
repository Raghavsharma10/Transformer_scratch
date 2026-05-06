def index(self):
        """
        Reset inspector buffers and index project sources dependencies.

        This have to be executed each time an event occurs.

        Note:
            If a Boussole exception occurs during operation, it will be catched
            and an error flag will be set to ``True`` so event operation will
            be blocked without blocking or breaking watchdog observer.
        """
        self._event_error = False

        try:
            compilable_files = self.finder.mirror_sources(
                self.settings.SOURCES_PATH,
                targetdir=self.settings.TARGET_PATH,
                excludes=self.settings.EXCLUDES
            )
            self.compilable_files = dict(compilable_files)
            self.source_files = self.compilable_files.keys()

            # Init inspector and do first inspect
            self.inspector.reset()
            self.inspector.inspect(
                *self.source_files,
                library_paths=self.settings.LIBRARY_PATHS
            )
        except BoussoleBaseException as e:
            self._event_error = True
            self.logger.error(six.text_type(e))