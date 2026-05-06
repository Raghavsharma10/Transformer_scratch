def rtm(  # type: ignore
        self, url: Optional[str] = None, bot_id: Optional[str] = None
    ) -> Iterator[events.Event]:
        """
        Iterate over event from the RTM API

        Args:
            url: Websocket connection url
            bot_id: Connecting bot ID

        Returns:
            :class:`slack.events.Event` or :class:`slack.events.Message`

        """
        while True:
            bot_id = bot_id or self._find_bot_id()
            url = url or self._find_rtm_url()
            for event in self._incoming_from_rtm(url, bot_id):
                yield event
            url = None