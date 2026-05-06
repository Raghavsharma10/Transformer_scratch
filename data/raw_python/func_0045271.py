async def get_matches(self, force_update=False) -> list:
        """ get all matches (once the tournament is started)

        |methcoro|

        Args:
            force_update (default=False): True to force an update to the Challonge API

        Returns:
            list[Match]:

        Raises:
            APIException

        """
        if force_update or self.matches is None:
            res = await self.connection('GET',
                                        'tournaments/{}/matches'.format(self._id),
                                        include_attachments=1)
            self._refresh_matches_from_json(res)
        return self.matches or []