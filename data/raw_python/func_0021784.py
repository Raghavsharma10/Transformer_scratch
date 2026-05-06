def dispatch(self, action: Action) -> Any:
        """
        Yields handlers matching the incoming :class:`slack.actions.Action` `callback_id`.

        Args:
            action: :class:`slack.actions.Action`

        Yields:
            handler
        """
        LOG.debug("Dispatching action %s, %s", action["type"], action["callback_id"])

        if action["type"] == "interactive_message":
            yield from self._dispatch_interactive_message(action)
        elif action["type"] in ("dialog_submission", "message_action"):
            yield from self._dispatch_action(action)
        else:
            raise UnknownActionType(action)