async def search_participant(self, name, force_update=False):
        """ search a participant by (display) name

        |methcoro|

        Args:
            name: display name of the participant
            force_update (dfault=False): True to force an update to the Challonge API

        Returns:
            Participant: None if not found

        Raises:
            APIException

        """
        if force_update or self.participants is None:
            await self.get_participants()
        if self.participants is not None:
            for p in self.participants:
                if p.name == name:
                    return p
        return None