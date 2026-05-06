async def get_next_match(self):
        """ Return the first open match found, or if none, the first pending match found

        |methcoro|

        Raises:
            APIException

        """
        if self._final_rank is not None:
            return None

        matches = await self.get_matches(MatchState.open_)

        if len(matches) == 0:
            matches = await self.get_matches(MatchState.pending)

        if len(matches) > 0:
            return matches[0]

        return None