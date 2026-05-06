def _create_user(self, username, password, mail, method, uuid):
        """Create a new user and all initial data"""

        try:
            if method == 'Invited':
                config_role = self.config.group_accept_invited
            else:
                config_role = self.config.group_accept_enrolled

            roles = []
            if ',' in config_role:
                for item in config_role.split(','):
                    roles.append(item.lstrip().rstrip())
            else:
                roles = [config_role]

            newuser = objectmodels['user']({
                'name': username,
                'passhash': std_hash(password, self.salt),
                'mail': mail,
                'uuid': std_uuid(),
                'roles': roles,
                'created': std_now()
            })

            if method == 'Invited':
                newuser.needs_password_change = True

            newuser.save()
        except Exception as e:
            self.log("Problem creating new user: ", type(e), e,
                     lvl=error)
            return

        try:
            newprofile = objectmodels['profile']({
                'uuid': std_uuid(),
                'owner': newuser.uuid
            })
            self.log("New profile uuid: ", newprofile.uuid,
                     lvl=verbose)

            newprofile.save()

            packet = {
                'component': 'hfos.enrol.enrolmanager',
                'action': 'enrol',
                'data': [True, mail]
            }
            self.fireEvent(send(uuid, packet))

            # TODO: Notify crew-admins
        except Exception as e:
            self.log("Problem creating new profile: ", type(e),
                     e, lvl=error)