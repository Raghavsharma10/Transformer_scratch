def reload(self, result=None):
        """
        Reloads the resource.
        """

        if result is None:
            result = self._client._get(
                self.__class__.base_url(
                    self.sys['space'].id,
                    self.sys['id'],
                    environment_id=self._environment_id
                )
            )

        self._update_from_resource(result)

        return self