def remove_observer(self, observer):
        """
        Stops thread and invalidates source.
        """
        super(PowerManagement, self).remove_observer(observer)
        if len(self._weak_observers) == 0:
            if not self._cf_run_loop:
                PowerManagement.notifications_observer.removeObserver(self)
            else:
                CFRunLoopSourceInvalidate(self._source)
                self._source = None