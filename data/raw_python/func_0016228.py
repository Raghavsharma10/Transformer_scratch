def set_active(self, username, active_state):
        """Set the active state of a user

        Args:
            username: The account username
            active_state: True or False

        Returns:
            True: If successful
            None: If no user or failure occurred
        """

        if active_state not in (True, False):
            raise ValueError("active_state must be True or False")

        user = self.get_user(username)
        if user is None:
            return None

        if user['active'] is active_state:
            # Already in desired state
            return True

        user['active'] = active_state
        response = self._put(self.rest_url + "/user",
                             params={"username": username},
                             data=json.dumps(user))

        if response.status_code == 204:
            return True

        return None