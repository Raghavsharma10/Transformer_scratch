async def change_file(self, file_path: str, description: str = None):
        """ change the file of that attachment

        |methcoro|

        Warning:
            |unstable|

        Args:
            file_path: path to the file you want to add / modify
            description: *optional* description for your attachment

        Raises:
            ValueError: file_path must not be None
            APIException

        """
        with open(file_path, 'rb') as f:
            await self._change(asset=f.read())