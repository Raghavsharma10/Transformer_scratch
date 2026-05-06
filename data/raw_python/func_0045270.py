async def get_match(self, m_id, force_update=False) -> Match:
        """ get a single match by id

        |methcoro|

        Args:
            m_id: match id
            force_update (default=False): True to force an update to the Challonge API

        Returns:
            Match

        Raises:
            APIException

        """
        found_m = self._find_match(m_id)
        if force_update or found_m is None:
            await self.get_matches()
            found_m = self._find_match(m_id)
        return found_m