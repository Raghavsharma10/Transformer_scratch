def update(self):
        """Get all events and process them by calling update_on_event()"""
        events = pygame.event.get()
        for e in events:
            self.update_on_event(e)

        for wid, cond in self._widgets:
            if cond():
                wid.update(events)