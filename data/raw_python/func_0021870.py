def clone(self) -> "Event":
        """
        Clone the event

        Returns:
            :class:`slack.events.Event`

        """
        return self.__class__(copy.deepcopy(self.event), copy.deepcopy(self.metadata))