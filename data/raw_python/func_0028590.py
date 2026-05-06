def _refresh(self):
        """ refresh internal directory cache """
        log.debug('refreshing directory cache')
        self._users.update(list(self._user_gen()))
        self._channels.update(list(self._channel_gen()))