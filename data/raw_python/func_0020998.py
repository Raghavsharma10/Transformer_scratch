def create_folder(self, name, parent_folder_id=0):
        """Create a folder

        If the folder exists, a BoxError will be raised.

        Args:
            folder_id (int): Name of the folder.

            parent_folder_id (int): ID of the folder where to create the new one.

        Returns:
            dict. Response from Box.

        Raises:
            BoxError: An error response is returned from Box (status_code >= 400).

            BoxHttpResponseError: Response from Box is malformed.

            requests.exceptions.*: Any connection related problem.
        """
        return self.__request("POST", "folders",
                        data={ "name": name,
                               "parent": {"id": unicode(parent_folder_id)} })