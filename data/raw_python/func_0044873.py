async def destroy_attachment(self, a: Attachment):
        """ destroy a match attachment

        |methcoro|

        Args:
            a: the attachment you want to destroy

        Raises:
            APIException

        """
        await self.connection('DELETE', 'tournaments/{}/matches/{}/attachments/{}'.format(self._tournament_id, self._id, a._id))
        if a in self.attachments:
            self.attachments.remove(a)