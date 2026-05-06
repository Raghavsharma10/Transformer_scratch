def userlogin(self, event):
        """Provides the newly authenticated user with a backlog and general
        channel status information"""

        try:
            user_uuid = event.useruuid
            user = objectmodels['user'].find_one({'uuid': user_uuid})

            if user_uuid not in self.lastlogs:
                self.log('Setting up lastlog for a new user.', lvl=debug)
                lastlog = objectmodels['chatlastlog']({
                    'owner': user_uuid,
                    'uuid': std_uuid(),
                    'channels': {}
                })
                lastlog.save()
                self.lastlogs[user_uuid] = lastlog

            self.users[user_uuid] = user
            self.user_attention[user_uuid] = None
            self._send_status(user_uuid, event.clientuuid)
        except Exception as e:
            self.log('Error during chat setup of user:', e, type(e), exc=True)