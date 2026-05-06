async def report_winner(self, winner: Participant, scores_csv: str):
        """ report scores and give a winner

        |methcoro|

        Args:
            winner: :class:Participant instance
            scores_csv: Comma separated set/game scores with player 1 score first (e.g. "1-3,3-0,3-2")

        Raises:
            ValueError: scores_csv has a wrong format
            APIException

        """
        await self._report(scores_csv, winner._id)