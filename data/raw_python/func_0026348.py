def location(self):
        """Return a string uniquely identifying the event.

        This string can be used to find the event in the event store UI (cf. id
        attribute, which is the UUID that at time of writing doesn't let you
        easily find the event).
        """
        if self._location is None:
            self._location = "{}/{}-{}".format(
                self.stream,
                self.type,
                self.sequence,
            )
        return self._location