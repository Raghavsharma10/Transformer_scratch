async def get_matches(self, state: MatchState = MatchState.all_):
        """ Return the matches of the given state

        |methcoro|

        Args:
            state: see :class:`MatchState`

        Raises:
            APIException

        """
        matches = await self.connection('GET',
                                        'tournaments/{}/matches'.format(self._tournament_id),
                                        state=state.value,
                                        participant_id=self._id)
        # return [await self._tournament.get_match(m['match']['id']) for m in matches] 3.6 only...
        ms = []
        for m in matches:
            ms.append(await self._tournament.get_match(m['match']['id']))
        return ms