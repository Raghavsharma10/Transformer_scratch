def add_observer(self, observer):
        """
        Spawns thread or adds IOPSNotificationCreateRunLoopSource directly to provided cf_run_loop
        @see: __init__
        """
        super(PowerManagement, self).add_observer(observer)
        if len(self._weak_observers) == 1:
            if not self._cf_run_loop:
                PowerManagement.notifications_observer.addObserver(self)
            else:
                @objc.callbackFor(IOPSNotificationCreateRunLoopSource)
                def on_power_sources_change(context):
                    self.on_power_source_notification()

                self._source = IOPSNotificationCreateRunLoopSource(on_power_sources_change, None)
                CFRunLoopAddSource(self._cf_run_loop, self._source, kCFRunLoopDefaultMode)