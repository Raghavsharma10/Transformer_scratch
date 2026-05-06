def add_user_to_group(self, username, groupname, raise_on_error=False):
        """Add a user to a group
        :param username: The username to assign to the group
        :param groupname: The group name into which to assign the user
        :return: True on success, False on failure.
        """
        data = {
                'name': groupname,
        }
        response = self._post(self.rest_url + "/user/group/direct",
                              params={"username": username,},
                              data=json.dumps(data))

        if response.status_code == 201:
            return True

        if raise_on_error:
            raise RuntimeError(response.json()['message'])

        return False