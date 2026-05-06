def _gather_event_handlers(self):
        """
        Searches for the event handlers in the current microservice class.

        :return:
        """
        self._extract_event_handlers_from_container(self)
        for module in self.modules:
            self._extract_event_handlers_from_container(module)