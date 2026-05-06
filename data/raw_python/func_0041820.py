def _setup_source_and_destination(self):
        """instantiate the classes that implement the source and destination
        crash storage systems."""
        try:
            self.source = self.config.source.crashstorage_class(
                self.config.source,
                quit_check_callback=self.quit_check
            )
        except Exception:
            self.config.logger.critical(
                'Error in creating crash source',
                exc_info=True
            )
            raise
        try:
            self.destination = self.config.destination.crashstorage_class(
                self.config.destination,
                quit_check_callback=self.quit_check
            )
        except Exception:
            self.config.logger.critical(
                'Error in creating crash destination',
                exc_info=True
            )
            raise