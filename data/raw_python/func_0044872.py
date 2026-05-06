async def attach_url(self, url: str, description: str = None) -> Attachment:
        """ add an url as an attachment

        |methcoro|

        Args:
            url: url you want to add
            description: *optional* description for your attachment

        Returns:
            Attachment:

        Raises:
            ValueError: url must not be None
            APIException

        """
        return await self._attach(url=url, description=description)