async def attach_file(self, file_path: str, description: str = None) -> Attachment:
        """ add a file as an attachment

        |methcoro|

        Warning:
            |unstable|

        Args:
            file_path: path to the file you want to add
            description: *optional* description for your attachment

        Returns:
            Attachment:

        Raises:
            ValueError: file_path must not be None
            APIException

        """
        with open(file_path, 'rb') as f:
            return await self._attach(f.read(), description)