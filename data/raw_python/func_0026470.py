def all(self, event):
        """Return all known schemata to the requesting client"""

        self.log("Schemarequest for all schemata from",
                 event.user, lvl=debug)
        response = {
            'component': 'hfos.events.schemamanager',
            'action': 'all',
            'data': l10n_schemastore[event.client.language]
        }
        self.fireEvent(send(event.client.uuid, response))