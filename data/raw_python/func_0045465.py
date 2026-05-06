async def destroy_tournament(self, t: Tournament):
        """ completely removes a tournament from Challonge

        |methcoro|

        Note:
            |from_api| Deletes a tournament along with all its associated records. There is no undo, so use with care!

        Raises:
            APIException

        """
        await self.connection('DELETE', 'tournaments/{}'.format(t.id))
        if t in self.tournaments:
            self.tournaments.remove(t)