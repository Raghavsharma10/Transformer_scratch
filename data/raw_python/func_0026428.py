def changepassword(self, event):
        """An enrolled user wants to change their password"""

        old = event.data['old']
        new = event.data['new']
        uuid = event.user.uuid

        # TODO: Write email to notify user of password change

        user = objectmodels['user'].find_one({'uuid': uuid})
        if std_hash(old, self.salt) == user.passhash:
            user.passhash = std_hash(new, self.salt)
            user.save()

            packet = {
                'component': 'hfos.enrol.enrolmanager',
                'action': 'changepassword',
                'data': True
            }
            self.fireEvent(send(event.client.uuid, packet))
            self.log('Successfully changed password for user', uuid)
        else:
            packet = {
                'component': 'hfos.enrol.enrolmanager',
                'action': 'changepassword',
                'data': False
            }
            self.fireEvent(send(event.client.uuid, packet))
            self.log('User tried to change password without supplying old one', lvl=warn)