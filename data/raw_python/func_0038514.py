def dispatch(self, message):
        """
        dispatch
        """
        handlers = []
        for handler in self.handlers:
            if handler["method"] != message.method:
                continue

            handlers.append(handler)
        return handlers