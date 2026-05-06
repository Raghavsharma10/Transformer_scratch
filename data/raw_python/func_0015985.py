def onAIOCompletion(self):
        """
        Call when eventfd notified events are available.
        """
        event_count = self.eventfd.read()
        trace('eventfd reports %i events' % event_count)
        # Even though eventfd signaled activity, even though it may give us
        # some number of pending events, some events seem to have been already
        # processed (maybe during io_cancel call ?).
        # So do not trust eventfd value, and do not even trust that there must
        # be even one event to process.
        self._aio_context.getEvents(0)