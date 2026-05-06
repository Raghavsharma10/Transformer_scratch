def get_tagged(self, event):
        """Return a list of tagged objects for a schema"""
        self.log("Tagged objects request for", event.data, "from",
                 event.user, lvl=debug)
        if event.data in self.tags:
            tagged = self._get_tagged(event.data)

            response = {
                'component': 'hfos.events.schemamanager',
                'action': 'get',
                'data': tagged
            }
            self.fireEvent(send(event.client.uuid, response))
        else:
            self.log("Unavailable schema requested!", lvl=warn)