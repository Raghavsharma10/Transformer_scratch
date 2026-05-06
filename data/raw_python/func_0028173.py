def get_all_users(configuration=None, **kwargs):
        # type: (Optional[Configuration], Any) -> List['User']
        """Get all users in HDX

        Args:
            configuration (Optional[Configuration]): HDX configuration. Defaults to global configuration.
            **kwargs: See below
            q (str): Restrict to names containing a string. Defaults to all users.
            order_by (str): Field by which to sort - any user field or edits (number_of_edits). Defaults to 'name'.

        Returns:
            List[User]: List of all users in HDX
        """
        user = User(configuration=configuration)
        user['id'] = 'all users'  # only for error message if produced
        result = user._write_to_hdx('list', kwargs, 'id')
        users = list()
        if result:
            for userdict in result:
                user = User(userdict, configuration=configuration)
                users.append(user)
        else:
            logger.debug(result)
        return users