def _fail(self, event, message='Invalid credentials'):
        """Sends a failure message to the requesting client"""

        notification = {
            'component': 'auth',
            'action': 'fail',
            'data': message
        }

        ip = event.sock.getpeername()[0]

        self.failing_clients[ip] = event
        Timer(3, Event.create('notify_fail', event.clientuuid, notification, ip)).register(self)