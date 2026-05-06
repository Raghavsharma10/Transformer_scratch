def request_reset(self, event):
        """An anonymous client requests a password reset"""

        self.log('Password reset request received:', event.__dict__, lvl=hilight)

        user_object = objectmodels['user']

        email = event.data.get('email', None)
        email_user = None

        if email is not None and user_object.count({'mail': email}) > 0:
            email_user = user_object.find_one({'mail': email})

        if email_user is None:
            self._fail(event, msg="Mail address unknown")
            return