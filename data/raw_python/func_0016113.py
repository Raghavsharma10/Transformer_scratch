def event_stream(self, raw=False, event_types=None):
        """Polls event bus using /v2/events

        :param bool raw: if true, yield raw event text, else yield MarathonEvent object
        :param event_types: a list of event types to consume
        :type event_types: list[type] or list[str]
        :returns: iterator with events
        :rtype: iterator
        """

        ef = EventFactory()

        params = {
            'event_type': [
                EventFactory.class_to_event[et] if isinstance(
                    et, type) and issubclass(et, MarathonEvent) else et
                for et in event_types or []
            ]
        }

        for raw_message in self._do_sse_request('/v2/events', params=params):
            try:
                _data = raw_message.decode('utf8').split(':', 1)

                if _data[0] == 'data':
                    if raw:
                        yield _data[1]
                    else:
                        event_data = json.loads(_data[1].strip())
                        if 'eventType' not in event_data:
                            raise MarathonError('Invalid event data received.')
                        yield ef.process(event_data)
            except ValueError:
                raise MarathonError('Invalid event data received.')