async def login_user(self, password, **kwds):
        """
            This function handles the registration of the given user credentials in the database
        """
        # find the matching user with the given email
        user_data = (await self._get_matching_user(fields=list(kwds.keys()), **kwds))['data']
        try:
            # look for a matching entry in the local database
            passwordEntry = self.model.select().where(
                self.model.user == user_data[root_query()][0]['pk']
            )[0]
        # if we couldn't acess the id of the result
        except (KeyError, IndexError) as e:
            # yell loudly
            raise RuntimeError('Could not find matching registered user')


        # if the given password matches the stored hash
        if passwordEntry and passwordEntry.password == password:
            # the remote entry for the user
            user = user_data[root_query()][0]
            # then return a dictionary with the user and sessionToken
            return {
                'user': user,
                'sessionToken': self._user_session_token(user)
            }

        # otherwise the passwords don't match
        raise RuntimeError("Incorrect credentials")