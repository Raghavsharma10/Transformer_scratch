def get_subscriptions(self, nodes=[]):
        """
        Returns all the channels where (optionally specified) nodes are subscribed
        """
        if len(nodes) > 0:
            data = [node for node in self.get_clients() if node['name'] in nodes]
        else:
            data = self.get_clients()
        channels = []
        for client in data:
            if 'subscriptions' in client:
                if isinstance(client['subscriptions'], list):
                    for channel in client['subscriptions']:
                        if channel not in channels:
                            channels.append(channel)
                else:
                    if client['subscriptions'] not in channels:
                        channels.append(client['subscriptions'])
        return channels