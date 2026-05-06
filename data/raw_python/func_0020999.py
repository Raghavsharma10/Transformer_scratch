def delete_folder(self, folder_id, recursive=True):
        """Delete an existing folder

        Args:
            folder_id (int): ID of the folder to delete.
            recursive (bool): Delete all subfolder if True.

        Returns:
            dict. Response from Box.

        Raises:
            BoxError: An error response is returned from Box (status_code >= 400).

            BoxHttpResponseError: Response from Box is malformed.

            requests.exceptions.*: Any connection related problem.
        """
        return self.__request("DELETE", "folders/%s" % (folder_id, ),
                        querystring={'recursive': unicode(recursive).lower()})