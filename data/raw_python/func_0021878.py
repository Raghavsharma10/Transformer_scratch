def dispatch(self, message: Message) -> Iterator[Any]:
        """
        Yields handlers matching the routing of the incoming :class:`slack.events.Message`

        Args:
            message: :class:`slack.events.Message`

        Yields:
            handler
        """
        if "text" in message:
            text = message["text"] or ""
        elif "message" in message:
            text = message["message"].get("text", "")
        else:
            text = ""

        msg_subtype = message.get("subtype")

        for subtype, matchs in itertools.chain(
            self._routes[message["channel"]].items(), self._routes["*"].items()
        ):
            if msg_subtype == subtype or subtype is None:
                for match, endpoints in matchs.items():
                    if match.search(text):
                        yield from endpoints