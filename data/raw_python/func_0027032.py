def _did_receive_event(self, connection):
        """ Receive an event from connection """

        if not self._is_running:
            return

        if connection.has_timeouted:
            return

        response = connection.response
        data = None

        if response.status_code != 200:
            pushcenter_logger.error("[NURESTPushCenter]: Connection failure [%s] %s" % (response.status_code, response.errors))

        else:
            data = response.data

            if len(self._delegate_methods) > 0:
                for m in self._delegate_methods:
                    try:
                        m(data)
                    except Exception as exc:
                        pushcenter_logger.error("[NURESTPushCenter] Delegate method %s failed:\n%s" % (m, exc))
            elif data:
                events = data['events']
                self.nb_events_received += len(events)
                self.nb_push_received += 1

                pushcenter_logger.info("[NURESTPushCenter] Received Push #%s (total=%s, latest=%s)\n%s" % (self.nb_push_received, self.nb_events_received, len(events), json.dumps(events, indent=4)))
                self._last_events.extend(events)

        if self._is_running:
            uuid = None
            if data and 'uuid' in data:
                uuid = data['uuid']

            self._listen(uuid)