def get_folder_items(self, folder_id,
                            limit=100, offset=0, fields_list=None):
        """Get files and folders inside a given folder

        Args:
            folder_id (int): Where to get files and folders info.

            limit (int): The number of items to return.

            offset (int): The item at which to begin the response.

            fields_list (list): List of attributes to get. All attributes if None.

        Returns:
            dict. Response from Box.

        Raises:
            BoxError: An error response is returned from Box (status_code >= 400).

            BoxHttpResponseError: Response from Box is malformed.

            requests.exceptions.*: Any connection related problem.
        """
        qs = {  "limit": limit,
                "offset": offset }
        if fields_list:
            qs['fields'] = ','.join(fields_list)
        return self.__request("GET", "folders/%s/items" % (folder_id, ),
                        querystring=qs)