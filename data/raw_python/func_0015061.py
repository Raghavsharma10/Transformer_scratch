def on_power_source_notification(self):
        """
        Called in response to IOPSNotificationCreateRunLoopSource() event.
        """
        for weak_observer in self._weak_observers:
            observer = weak_observer()
            if observer:
                observer.on_power_sources_change(self)
                observer.on_time_remaining_change(self)