def get_or_create_user(self, username, password):
        '''
        Get or create the given user
        '''

        # Get the groups for this user
        info = self.get_ad_info(username, password)
        self.debug("INFO found: {}".format(info))

        # Find the user
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            user = User(username=username)

        # Update user
        user.first_name = info.get('first_name', '')
        user.last_name = info.get('last_name', '')
        user.email = info.get('email', '')

        # Check if the user is in the Administrators groups
        is_admin = False
        for domain in info['groups']:
            if 'Domain Admins' in info['groups'][domain]:
                is_admin = True
                break

        # Set the user permissions
        user.is_staff = is_admin
        user.is_superuser = is_admin

        # Refresh the password
        user.set_password(password)

        # Validate the selected user and gotten information
        user = self.validate(user, info)
        if user:
            self.debug("User got validated!")

            # Autosave the user until this point
            user.save()

            # Synchronize user
            self.synchronize(user, info)
        else:
            self.debug("User didn't pass validation!")

        # Finally return user
        return user