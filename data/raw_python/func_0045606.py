async def change_url(self, url: str, description: str = None):
        """ change the url of that attachment

        |methcoro|

        Args:
            url: url you want to change
            description: *optional* description for your attachment

        Raises:
            ValueError: url must not be None
            APIException

        """
        await self._change(url=url, description=description)