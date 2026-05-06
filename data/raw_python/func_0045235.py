async def get_next_opponent(self):
        """ Get the opponent of the potential next match. See :func:`get_next_match`

        |methcoro|

        Raises:
            APIException

        """
        next_match = await self.get_next_match()
        if next_match is not None:
            opponent_id = next_match.player1_id if next_match.player2_id == self._id else next_match.player2_id
            return await self._tournament.get_participant(opponent_id)
        return None