def on_moved(self, event):
        """
        A move event is just proxied to an on_deleted call followed by
        an on_created call.
        """
        self.on_deleted(events.FileDeletedEvent(event.src_path))
        self.on_created(events.FileCreatedEvent(event.dest_path))