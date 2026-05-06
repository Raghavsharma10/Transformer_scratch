def add_update_user(self, user, capacity=None):
        # type: (Union[hdx.data.user.User,Dict,str],Optional[str]) -> None
        """Add new or update existing user in organization with new metadata. Capacity eg. member, admin
        must be supplied either within the User object or dictionary or using the capacity argument (which takes
        precedence).

        Args:
            user (Union[User,Dict,str]): Either a user id or user metadata either from a User object or a dictionary
            capacity (Optional[str]): Capacity of user eg. member, admin. Defaults to None.

        Returns:
            None

        """
        if isinstance(user, str):
            user = hdx.data.user.User.read_from_hdx(user, configuration=self.configuration)
        elif isinstance(user, dict):
            user = hdx.data.user.User(user, configuration=self.configuration)
        if isinstance(user, hdx.data.user.User):
            users = self.data.get('users')
            if users is None:
                users = list()
                self.data['users'] = users
            if capacity is not None:
                user['capacity'] = capacity
            self._addupdate_hdxobject(users, 'name', user)
            return
        raise HDXError('Type %s cannot be added as a user!' % type(user).__name__)