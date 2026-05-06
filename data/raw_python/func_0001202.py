def notify_prepared(self, args=None, kwargs=None, **opts):
        """Like notify allows to pass more options to the underlying
        `Signal.prepare_notification()` method.

        The allowed options are:

        notify_external : bool
          a flag indicating if the notification should also include the
          registered `~.external.ExternalSignaller` in the notification. It's
          ``True`` by default

        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        loop = kwargs.pop('loop', self.loop)
        return self.signal.prepare_notification(
            subscribers=self.subscribers, instance=self.instance,
            loop=loop, **opts).run(*args, **kwargs)