def get(self):
        """
        Retrieve the current configured SharedStorages entries
        :return: [list] List containing the current SharedStorages entries
        """
        request = self._call(GetSharedStorages)
        response = request.commit()
        return response['Value']