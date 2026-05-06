def remove_user_from_group(self, username, groupname, raise_on_error=False):
        """Remove a user from a group

	Attempts to remove a user from a group

	Args
             username: The username to remove from the group.
             groupname: The group name to be removed from the user.

        Returns:
            True: Succeeded
            False: If unsuccessful
        """

        response = self._delete(self.rest_url + "/group/user/direct",params={"username": username, "groupname": groupname})

        if response.status_code == 204:
            return True

        if raise_on_error:
            raise RuntimeError(response.json()['message'])

        return False