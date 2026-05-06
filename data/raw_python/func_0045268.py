async def add_participant(self, display_name: str = None, username: str = None, email: str = None, seed: int = 0, misc: str = None, **params):
        """ add a participant to the tournament

        |methcoro|

        Args:
            display_name: The name displayed in the bracket/schedule - not required if email or challonge_username is provided. Must be unique per tournament.
            username: Provide this if the participant has a Challonge account. He or she will be invited to the tournament.
            email: Providing this will first search for a matching Challonge account. If one is found, this will have the same effect as the "challonge_username" attribute. If one is not found, the "new-user-email" attribute will be set, and the user will be invited via email to create an account.
            seed: The participant's new seed. Must be between 1 and the current number of participants (including the new record). Overwriting an existing seed will automatically bump other participants as you would expect.
            misc: Max: 255 characters. Multi-purpose field that is only visible via the API and handy for site integration (e.g. key to your users table)
            params: optional params (see http://api.challonge.com/v1/documents/participants/create)

        Returns:
            Participant: newly created participant

        Raises:
            APIException

        """
        assert_or_raise((display_name is None) ^ (username is None),
                        ValueError,
                        'One of display_name or username must not be None')

        params.update({
            'name': display_name or '',
            'challonge_username': username or '',
        })
        if email is not None:
            params.update({'email': email})
        if seed != 0:
            params.update({'seed': seed})
        if misc is not None:
            params.update({'misc': misc})
        res = await self.connection('POST',
                                    'tournaments/{}/participants'.format(self._id),
                                    'participant',
                                    **params)
        new_p = self._create_participant(res)
        self._add_participant(new_p)
        return new_p