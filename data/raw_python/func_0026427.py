def change(self, event):
        """An admin user requests a change to an enrolment"""

        uuid = event.data['uuid']
        status = event.data['status']

        if status not in ['Open', 'Pending', 'Accepted', 'Denied', 'Resend']:
            self.log('Erroneous status for enrollment requested!', lvl=warn)
            return

        self.log('Changing status of an enrollment', uuid, 'to', status)

        enrollment = objectmodels['enrollment'].find_one({'uuid': uuid})
        if enrollment is not None:
            self.log('Enrollment found', lvl=debug)
        else:
            return

        if status == 'Resend':
            enrollment.timestamp = std_now()
            enrollment.save()
            self._send_invitation(enrollment, event)
            reply = {True: 'Resent'}
        else:
            enrollment.status = status
            enrollment.save()
            reply = {True: enrollment.serializablefields()}

        if status == 'Accepted' and enrollment.method == 'Enrolled':
            self._create_user(enrollment.name, enrollment.password, enrollment.email, 'Invited', event.client.uuid)
            self._send_acceptance(enrollment, None, event)

        packet = {
            'component': 'hfos.enrol.enrolmanager',
            'action': 'change',
            'data': reply
        }
        self.log('packet:', packet, lvl=verbose)
        self.fireEvent(send(event.client.uuid, packet))
        self.log('Enrollment changed', lvl=debug)