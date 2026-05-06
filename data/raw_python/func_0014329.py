def unpublish(self):
        """
        Unpublishes the resource.
        """

        self._client._delete(
            "{0}/published".format(
                self.__class__.base_url(
                    self.sys['space'].id,
                    self.sys['id'],
                    environment_id=self._environment_id
                ),
            ),
            headers=self._update_headers()
        )

        return self.reload()