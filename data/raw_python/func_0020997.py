def find_id_in_folder(self, name, parent_folder_id=0):
        """Find a folder or a file ID from its name, inside a given folder.

        Args:
            name (str): Name of the folder or the file to find.

            parent_folder_id (int): ID of the folder where to search.

        Returns:
            int. ID of the file or folder found. None if not found.

        Raises:
            BoxError: An error response is returned from Box (status_code >= 400).

            BoxHttpResponseError: Response from Box is malformed.

            requests.exceptions.*: Any connection related problem.
        """
        if name is None or len(name) == 0:
            return parent_folder_id
        offset = 0
        resp = self.get_folder_items(parent_folder_id,
                                     limit=1000, offset=offset,
                                     fields_list=['name'])
        total = int(resp['total_count'])
        while offset < total:
            found = self.__find_name(resp, name)
            if found is not None:
                return found
            offset += int(len(resp['entries']))
            resp = self.get_folder_items(parent_folder_id,
                                            limit=1000, offset=offset,
                                            fields_list=['name'])

        return None