def authenticate(self, *args, **kwargs):
        '''
        Authenticate the user agains LDAP
        '''

        # Get config
        username = kwargs.get("username", None)
        password = kwargs.get("password", None)

        # Check user in Active Directory (authorization == None if can not connect to Active Directory Server)
        authorization = self.ldap_link(username, password, mode='LOGIN')

        if authorization:
            # The user was validated in Active Directory
            user = self.get_or_create_user(username, password)
            # Get or get_create_user will revalidate the new user
            if user:
                # If the user has been properly validated
                user.is_active = True
                user.save()
        else:
            # Locate user in our system
            user = User.objects.filter(username=username).first()
            if user and not user.is_staff:
                # If access was denied
                if authorization is False or getattr(settings, "AD_LOCK_UNAUTHORIZED", False):
                    # Deactivate the user
                    user.is_active = False
                    user.save()

            # No access and no user here
            user = None

        # Return the final decision
        return user