def delete(self):
        """
        Deletes the space
        """

        return self._client._delete(
            self.__class__.base_url(
                self.sys['id']
            )
        )