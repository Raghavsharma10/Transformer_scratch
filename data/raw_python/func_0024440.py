def is_auth(self):
        """
        A property that indicates if current user is logged in or not.

        Returns:
            Boolean.
        """
        if self.user_id is None:
            self.user_id = self.session.get('user_id')
        return bool(self.user_id)