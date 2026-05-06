def unsubscribe(self, event):
        """Unsubscribe from an object's future changes"""
        # TODO: Automatic Unsubscription
        uuids = event.data

        if not isinstance(uuids, list):
            uuids = [uuids]

        result = []

        for uuid in uuids:
            if uuid in self.subscriptions:
                self.subscriptions[uuid].pop(event.client.uuid)

                if len(self.subscriptions[uuid]) == 0:
                    del (self.subscriptions[uuid])

                result.append(uuid)

        result = {
            'component': 'hfos.events.objectmanager',
            'action': 'unsubscribe',
            'data': {
                'uuid': result, 'success': True
            }
        }

        self._respond(None, result, event)