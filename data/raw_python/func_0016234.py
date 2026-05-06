def get_nested_groups(self, username):
        """Retrieve a list of all group names that have <username> as a direct or indirect member.

        Args:
            username: The account username.


        Returns:
            list:
                A list of strings of group names.
        """

        response = self._get(self.rest_url + "/user/group/nested",
                             params={"username": username})

        if not response.ok:
            return None

        return [g['name'] for g in response.json()['groups']]