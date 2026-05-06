def archive(self):
        """
        Archives the resource.
        """

        self._client._put(
            "{0}/archived".format(
                self.__class__.base_url(
                    self.sys['space'].id,
                    self.sys['id'],
                    environment_id=self._environment_id
                ),
            ),
            {},
            headers=self._update_headers()
        )

        return self.reload()