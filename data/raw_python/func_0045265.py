async def get_participant(self, p_id: int, force_update=False) -> Participant:
        """ get a participant by its id

        |methcoro|

        Args:
            p_id: participant id
            force_update (dfault=False): True to force an update to the Challonge API

        Returns:
            Participant: None if not found

        Raises:
            APIException

        """
        found_p = self._find_participant(p_id)
        if force_update or found_p is None:
            await self.get_participants()
            found_p = self._find_participant(p_id)
        return found_p