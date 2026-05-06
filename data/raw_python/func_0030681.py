def add(self, *args, **kwargs):
        """Add a new record to the section"""

        if self.start and self.start.state == 'done' and kwargs.get('log_action') != 'done':
            raise ProgressLoggingError("Can't add -- process section is done")

        self.augment_args(args, kwargs)

        kwargs['log_action'] = kwargs.get('log_action', 'add')

        rec = Process(**kwargs)

        self._session.add(rec)

        self.rec = rec

        if self._logger:
            self._logger.info(self.rec.log_str)

        self._session.commit()
        self._ai_rec_id = None

        return self.rec.id