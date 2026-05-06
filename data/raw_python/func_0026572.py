def client_details(self, *args):
        """Display known details about a given client"""

        self.log(_('Client details:', lang='de'))
        client = self._clients[args[0]]

        self.log('UUID:', client.uuid, 'IP:', client.ip, 'Name:', client.name, 'User:', self._users[client.useruuid],
                 pretty=True)