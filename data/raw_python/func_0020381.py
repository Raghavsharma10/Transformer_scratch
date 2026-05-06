def is_logged(self, user):
        """Check if a logged user is trying to access the register page.
           If so, redirect him/her to his/her profile"""

        response = None
        if user.is_authenticated():
            if not user.needs_update:
                response = redirect('user_profile', username=user.username)

        return response