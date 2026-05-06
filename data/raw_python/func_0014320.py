def delete(self):
        """
        Deletes the resource.
        """

        return self._client._delete(
            self.__class__.base_url(
                self.sys['space'].id,
                self.sys['id'],
                environment_id=self._environment_id
            )
        )