def login_user(self, request):
        """
        Try to login user by net identity.
        Do nothing in case of failure.
        """
        # only actavted users can login if activation required.
        user = auth.authenticate(identity=self.identity, provider=self.provider)
        if user and settings.ACTIVATION_REQUIRED and not user.is_active:
            messages.add_message(request, messages.ERROR, lang.NOT_ACTIVATED)
            return redirect(settings.ACTIVATION_REDIRECT_URL)
        # login
        if user:
            auth.login(request, user)
            return True

        return False