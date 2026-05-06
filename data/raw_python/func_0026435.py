def _invite(self, name, method, email, uuid, event, password=""):
        """Actually invite a given user"""

        props = {
            'uuid': std_uuid(),
            'status': 'Open',
            'name': name,
            'method': method,
            'email': email,
            'password': password,
            'timestamp': std_now()
        }
        enrollment = objectmodels['enrollment'](props)
        enrollment.save()

        self.log('Enrollment stored', lvl=debug)

        self._send_invitation(enrollment, event)

        packet = {
            'component': 'hfos.enrol.enrolmanager',
            'action': 'invite',
            'data': [True, email]
        }
        self.fireEvent(send(uuid, packet))