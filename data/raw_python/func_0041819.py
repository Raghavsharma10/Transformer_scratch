def _transform(self, crash_id):
        """this default transform function only transfers raw data from the
        source to the destination without changing the data.  While this may
        be good enough for the raw crashmover, the processor would override
        this method to create and save processed crashes"""
        try:
            raw_crash = self.source.get_raw_crash(crash_id)
        except Exception as x:
            self.config.logger.error(
                "reading raw_crash: %s",
                str(x),
                exc_info=True
            )
            raw_crash = {}
        try:
            dumps = self.source.get_raw_dumps(crash_id)
        except Exception as x:
            self.config.logger.error(
                "reading dump: %s",
                str(x),
                exc_info=True
            )
            dumps = {}
        try:
            self.destination.save_raw_crash(raw_crash, dumps, crash_id)
            self.config.logger.info('saved - %s', crash_id)
        except Exception as x:
            self.config.logger.error(
                "writing raw: %s",
                str(x),
                exc_info=True
            )
        else:
            try:
                self.source.remove(crash_id)
            except Exception as x:
                self.config.logger.error(
                    "removing raw: %s",
                    str(x),
                    exc_info=True
                )