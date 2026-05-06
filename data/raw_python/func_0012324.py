async def register_user(self, password, **kwds):
        """
            This function is used to provide a sessionToken for later requests.

            Args:
                uid (str): The
        """
        # so make one
        user = await self._create_remote_user(password=password, **kwds)
        # if there is no pk field
        if not 'pk' in user:
            # make sure the user has a pk field
            user['pk'] = user['id']

        # the query to find a matching query
        match_query = self.model.user == user['id']

        # if the user has already been registered
        if self.model.select().where(match_query).count() > 0:
            # yell loudly
            raise RuntimeError('The user is already registered.')

        # create an entry in the user password table
        password = self.model(user=user['id'], password=password)

        # save it to the database
        password.save()

        # return a dictionary with the user we created and a session token for later use
        return {
            'user': user,
            'sessionToken': self._user_session_token(user)
        }