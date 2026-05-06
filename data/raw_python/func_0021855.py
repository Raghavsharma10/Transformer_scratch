async def _incoming_from_rtm(
        self, url: str, bot_id: str
    ) -> AsyncIterator[events.Event]:
        """
        Connect and discard incoming RTM event if necessary.

        :param url: Websocket url
        :param bot_id: Bot ID
        :return: Incoming events
        """
        async for data in self._rtm(url):
            event = events.Event.from_rtm(json.loads(data))
            if sansio.need_reconnect(event):
                break
            elif sansio.discard_event(event, bot_id):
                continue
            else:
                yield event