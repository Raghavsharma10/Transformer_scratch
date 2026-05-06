def userlogin(self, event):
        """Checks if an alert is ongoing and alerts the newly connected
        client, if so."""

        client_uuid = event.clientuuid

        self.log(event.user, pretty=True, lvl=verbose)

        self.log('Adding client')
        self.clients[event.clientuuid] = event.user

        for topic, alert in self.alerts.items():
            self.alert(client_uuid, alert)