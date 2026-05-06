def add_update_users(self, users, capacity=None):
        # type: (List[Union[hdx.data.user.User,Dict,str]],Optional[str]) -> None
        """Add new or update existing users in organization with new metadata. Capacity eg. member, admin
        must be supplied either within the User object or dictionary or using the capacity argument (which takes
        precedence).

        Args:
            users (List[Union[User,Dict,str]]): A list of either user ids or users metadata from User objects or dictionaries
            capacity (Optional[str]): Capacity of users eg. member, admin. Defaults to None.

        Returns:
            None
        """
        if not isinstance(users, list):
            raise HDXError('Users should be a list!')
        for user in users:
            self.add_update_user(user, capacity)