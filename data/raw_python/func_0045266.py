async def get_participants(self, force_update=False) -> list:
        """ get all participants

        |methcoro|

        Args:
            force_update (default=False): True to force an update to the Challonge API

        Returns:
            list[Participant]:

        Raises:
            APIException

        """
        if force_update or self.participants is None:
            res = await self.connection('GET', 'tournaments/{}/participants'.format(self._id))
            self._refresh_participants_from_json(res)
        return self.participants or []