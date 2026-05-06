def accept(self, event):
        """A challenge/response for an enrolment has been accepted"""

        self.log('Invitation accepted:', event.__dict__, lvl=debug)
        try:
            uuid = event.data
            enrollment = objectmodels['enrollment'].find_one({
                'uuid': uuid
            })

            if enrollment is not None:
                self.log('Enrollment found', lvl=debug)
                if enrollment.status == 'Open':
                    self.log('Enrollment is still open', lvl=debug)
                    if enrollment.method == 'Invited' and self.config.auto_accept_invited:
                        enrollment.status = 'Accepted'

                        data = 'You should have received an email with your new password ' \
                               'and can now log in to the system and start to use it. <br/>' \
                               'Please change your password immediately after logging in'
                        password = std_human_uid().replace(" ", '')

                        self._create_user(enrollment.name, password, enrollment.email, enrollment.method, uuid)
                        self._send_acceptance(enrollment, password, event)
                    elif enrollment.method == 'Enrolled' and self.config.auto_accept_enrolled:
                        enrollment.status = 'Accepted'
                        data = 'Your account is now activated.'

                        self._create_user(enrollment.name, enrollment.password, enrollment.email, enrollment.method,
                                          uuid)

                        # TODO: Evaluate if sending an acceptance mail makes sense
                        # self._send_acceptance(enrollment, "", event)
                    else:
                        enrollment.status = 'Pending'
                        data = 'Someone has to confirm your enrollment ' \
                               'first. Thank you, for your patience.'
                        # TODO: Alert admin users
                    enrollment.save()

                # Reaffirm acceptance to end user, when clicking on the link multiple times
                elif enrollment.status == 'Accepted':
                    data = 'You can now log in to the system and start to use it.'
                elif enrollment.status == 'Pending':
                    data = 'Someone has to confirm your enrollment ' \
                           'first. Thank you, for your patience.'
                else:
                    self.log('Enrollment has been closed already!', lvl=warn)
                    self._fail(event)
                    return
                packet = {
                    'component': 'hfos.enrol.enrolmanager',
                    'action': 'accept',
                    'data': {True: data}
                }
                self.fireEvent(send(event.client.uuid, packet))
            else:
                self.log('No enrollment available.', lvl=warn)
                self._fail(event)
        except Exception as e:
            self.log('Error during invitation accept handling:', e, type(e),
                     lvl=warn, exc=True)