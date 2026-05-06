def update(self, *args, **kwargs):
        """Update the last section record"""

        self.augment_args(args, kwargs)

        kwargs['log_action'] = kwargs.get('log_action', 'update')

        if not self.rec:
            return self.add(**kwargs)
        else:
            for k, v in kwargs.items():

                # Don't update object; use whatever was set in the original record
                if k not in ('source', 's_vid', 'table', 't_vid', 'partition', 'p_vid'):
                    setattr(self.rec, k, v)

            self._session.merge(self.rec)
            if self._logger:
                self._logger.info(self.rec.log_str)
            self._session.commit()

            self._ai_rec_id = None
            return self.rec.id