def remove_logger(self, cb_id):
        '''Remove a logger.

        @param cb_id The ID of the logger to remove.
        @raises NoLoggerError

        '''
        if cb_id not in self._loggers:
            raise exceptions.NoLoggerError(cb_id, self.name)
        conf = self.object.get_configuration()
        res = conf.remove_service_profile(cb_id.get_bytes())
        del self._loggers[cb_id]