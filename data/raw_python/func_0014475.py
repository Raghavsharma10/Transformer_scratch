def set_event(self, event):
        """
        Set an event on this buffer.  When data is ready to be read (or the
        buffer has been closed), the event will be set.  When no data is
        ready, the event will be cleared.
        
        @param event: the event to set/clear
        @type event: Event
        """
        self._event = event
        if len(self._buffer) > 0:
            event.set()
        else:
            event.clear()