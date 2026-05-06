def get_users(self, capacity=None):
        # type: (Optional[str]) -> List[User]
        """Returns the organization's users.

        Args:
            capacity (Optional[str]): Filter by capacity eg. member, admin. Defaults to None.
        Returns:
            List[User]: Organization's users.
        """
        users = list()
        usersdicts = self.data.get('users')
        if usersdicts is not None:
            for userdata in usersdicts:
                if capacity is not None and userdata['capacity'] != capacity:
                    continue
                id = userdata.get('id')
                if id is None:
                    id = userdata['name']
                user = hdx.data.user.User.read_from_hdx(id, configuration=self.configuration)
                user['capacity'] = userdata['capacity']
                users.append(user)
        return users