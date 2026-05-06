async def remove_participant(self, p: Participant):
        """ remove a participant from the tournament

        |methcoro|

        Args:
            p: the participant to remove

        Raises:
            APIException

        """
        await self.connection('DELETE', 'tournaments/{}/participants/{}'.format(self._id, p._id))
        if p in self.participants:
            self.participants.remove(p)