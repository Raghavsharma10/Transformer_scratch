async def change_votes(self, player1_votes: int = None, player2_votes: int = None, add: bool = False):
        """ change the votes for either player

        |methcoro|
        The votes will be overriden by default,
        If `add` is set to True, another API request call will be made to ensure the local is up to date with
        the Challonge server. Then the votes given in argument will be added to those on the server

        Args:
            player1_votes: if set, the player 1 votes will be changed to this value, or added to the current value if `add` is set
            player1_votes: if set, the player 2 votes will be changed to this value, or added to the current value if `add` is set
            add: if set, votes in parameters will be added instead of overriden

        Raises:
            ValueError: one of the votes arguments must not be None
            APIException

        """
        assert_or_raise(player1_votes is not None or player2_votes is not None,
                        ValueError,
                        'One of the votes must not be None')

        if add:
            # order a fresh update of this match
            res = await self.connection('GET', 'tournaments/{}/matches/{}'.format(self._tournament_id, self._id))
            self._refresh_from_json(res)
            if player1_votes is not None:
                player1_votes += self._player1_votes or 0
            if player2_votes is not None:
                player2_votes += self._player2_votes or 0

        params = {}
        if player1_votes is not None:
                params.update({'player1_votes': player1_votes})
        if player2_votes is not None:
                params.update({'player2_votes': player2_votes})
        res = await self.connection('PUT',
                                    'tournaments/{}/matches/{}'.format(self._tournament_id, self._id),
                                    'match',
                                    **params)
        self._refresh_from_json(res)