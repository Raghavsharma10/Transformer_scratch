def delete(self):
        """
        Deletes the object

        :return:
        :rtype: None
        """
        return self._delete_request(endpoint=self.ENDPOINT + '/' + str(self.id))