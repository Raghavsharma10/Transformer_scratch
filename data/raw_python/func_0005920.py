def set_client_params(
            self, start_unsubscribed=None, clear_on_exit=None, unsubscribe_on_reload=None,
            announce_interval=None):
        """Sets subscribers related params.

        :param bool start_unsubscribed: Configure subscriptions but do not send them.
            .. note:: Useful with master FIFO.

        :param bool clear_on_exit: Force clear instead of unsubscribe during shutdown.

        :param bool unsubscribe_on_reload: Force unsubscribe request even during graceful reload.

        :param int announce_interval: Send subscription announce at the specified interval. Default: 10 master cycles.

        """
        self._set('start-unsubscribed', start_unsubscribed, cast=bool)
        self._set('subscription-clear-on-shutdown', clear_on_exit, cast=bool)
        self._set('unsubscribe-on-graceful-reload', unsubscribe_on_reload, cast=bool)
        self._set('subscribe-freq', announce_interval)

        return self._section