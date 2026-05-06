def status(self, event):
        """An anonymous client wants to know if we're open for enrollment"""

        self.log('Registration status requested')

        response = {
            'component': 'hfos.enrol.enrolmanager',
            'action': 'status',
            'data': self.config.allow_registration
        }

        self.fire(send(event.client.uuid, response))