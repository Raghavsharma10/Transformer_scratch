def ping(self, event):
        """Perform a ping to measure client <-> node latency"""

        self.log('Client ping received:', event.data, lvl=verbose)
        response = {
            'component': 'hfos.ui.clientmanager',
            'action': 'pong',
            'data': [event.data, time() * 1000]
        }

        self.fire(send(event.client.uuid, response))