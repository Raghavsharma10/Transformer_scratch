def subscribe(self, event):
        """Subscribe to an object's future changes"""
        uuids = event.data

        if not isinstance(uuids, list):
            uuids = [uuids]

        subscribed = []
        for uuid in uuids:
            try:
                self._add_subscription(uuid, event)
                subscribed.append(uuid)
            except KeyError:
                continue

        result = {
            'component': 'hfos.events.objectmanager',
            'action': 'subscribe',
            'data': {
                'uuid': subscribed, 'success': True
            }
        }
        self._respond(None, result, event)