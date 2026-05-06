def register(self, callback_id: str, handler: Any, name: str = "*") -> None:
        """
        Register a new handler for a specific :class:`slack.actions.Action` `callback_id`.
        Optional routing based on the action name too.

        The name argument is useful for actions of type `interactive_message` to provide
        a different handler for each individual action.

        Args:
            callback_id: Callback_id the handler is interested in
            handler: Callback
            name: Name of the action (optional).
        """
        LOG.info("Registering %s, %s to %s", callback_id, name, handler)
        if name not in self._routes[callback_id]:
            self._routes[callback_id][name] = []

        self._routes[callback_id][name].append(handler)