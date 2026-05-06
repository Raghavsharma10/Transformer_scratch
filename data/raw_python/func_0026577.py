def who(self, *args):
        """Display a table of connected users and clients"""
        if len(self._users) == 0:
            self.log('No users connected')
            if len(self._clients) == 0:
                self.log('No clients connected')
                return

        Row = namedtuple("Row", ['User', 'Client', 'IP'])
        rows = []

        for user in self._users.values():
            for key, client in self._clients.items():
                if client.useruuid == user.uuid:
                    row = Row(user.account.name, key, client.ip)
                    rows.append(row)

        for key, client in self._clients.items():
            if client.useruuid is None:
                row = Row('ANON', key, client.ip)
                rows.append(row)

        self.log("\n" + std_table(rows))