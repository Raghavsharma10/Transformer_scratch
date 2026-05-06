def get_subscriptions_channel(self, search_channel):
        """
        Return all the nodes that are subscribed to the specified channel
        """
        data = self.get_clients()
        clients = []
        for client in data:
            if 'subscriptions' in client:
                if isinstance(client['subscriptions'], list):
                    if search_channel in client['subscriptions']:
                        clients.append(client['name'])
                else:
                    if search_channel == client['subscriptions']:
                        clients.append(client['name'])
        return clients