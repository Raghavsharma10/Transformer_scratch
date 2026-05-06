def copy_file(self, file_id, dest_folder_id):
        """Copy file to new destination

        Args:
            file_id (int): ID of the folder.

            dest_folder_id (int): ID of parent folder you are copying to.

        Returns:
            dict. Response from Box.

        Raises:
            BoxError: An error response is returned from Box (status_code >= 400).

            BoxError: 409 - Item with the same name already exists.
            In this case you will need download the file and upload a new version to your destination.
            (Box currently doesn't have a method to copy a new verison.)

            BoxHttpResponseError: Response from Box is malformed.

            requests.exceptions.*: Any connection related problem.
        """

        return self.__request("POST", "/files/" + unicode(file_id) + "/copy",
                        data={ "parent": {"id": unicode(dest_folder_id)} })